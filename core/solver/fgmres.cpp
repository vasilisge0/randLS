#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../../cuda/solver/lsqr_kernels.cuh"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../utils/io.hpp"
#include "../blas/blas.hpp"
#include "../dense/dense.hpp"
#include "../memory/memory.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "fgmres.hpp"
#include "solver.hpp"
#include "base_types.hpp"


namespace rls {
namespace solver {
namespace fgmres {


template <typename value_type, typename index_type>
void givens_qr(dim2 size, value_type* hessenberg_mtx,
               index_type ld_h,
               value_type* rhs, index_type cur_iter, index_type max_iter,
               std::vector<std::pair<value_type, value_type>>& givens_cache)
{
    for (index_type row = 0; row < cur_iter + 1; row++) {
        // Compute givens rotation matrix.
        if (row == cur_iter) {
            auto alpha = hessenberg_mtx[row + ld_h*cur_iter];
            auto beta = hessenberg_mtx[(row + 1) + ld_h*cur_iter];
            auto r = std::hypot(alpha, beta);
            auto c =   alpha / r;
            auto s = - beta  / r;
            givens_cache[row].first = c;
            givens_cache[row].second = s;
            // update rhs
            auto tmp_rhs_0 = rhs[row];
            auto tmp_rhs_1 = rhs[row + 1];
            rhs[row]     = c*tmp_rhs_0 - s*tmp_rhs_1;
            rhs[row + 1] = s*tmp_rhs_0 + c*tmp_rhs_1;
        }
        auto c = givens_cache[row].first;
        auto s = givens_cache[row].second;
        // Update hessenberg matrix.
        for (index_type col = cur_iter; col < cur_iter + 1; col++) {
            const auto h0 = hessenberg_mtx[row +     ld_h * col];
            const auto h1 = hessenberg_mtx[row + 1 + ld_h * col];
            hessenberg_mtx[row +     ld_h*col] = c*h0 - s*h1;
            hessenberg_mtx[row + 1 + ld_h*col] = s*h0 + c*h1;
        }
    }
}

// size is the dimension of mtx
// The dimension of the global system is global_length = size[0] + size[1].
template <typename value_type_in, typename value_type, typename index_type, ContextType device_type>
void initialize(
    dim2 size, value_type* mtx, value_type* sol, value_type* rhs,
    preconditioner::preconditioner<value_type_in, value_type, index_type>*
    precond,
    fgmres::temp_scalars<value_type, index_type>* scalars,
    fgmres::temp_vectors<value_type_in, value_type, index_type, device_type>* vectors,
    magma_queue_t& queue)
{
    auto global_length = size[0] + size[1];
    index_type inc_v = 1;
    blas::copy(global_length, rhs, inc_v, vectors->residual, inc_v, queue);

    fgmres::gemv(MagmaNoTrans, size[0], size[1], -1.0, mtx, size[0], sol,
                 vectors->inc, 1.0, vectors->residual, vectors->inc,
                 vectors->temp, queue);


    // Left apply.
    precond->apply(MagmaTrans, vectors->residual, vectors->inc);

    scalars->beta =
        blas::norm2(global_length, vectors->residual, vectors->inc, queue);

    vectors->tmp_cpu->get_values()[0] = scalars->beta;
    auto max_iter = vectors->max_iter_;
    for (auto i = 0; i < max_iter + 1; i++) {
        vectors->hessenberg_rhs[i] = 0;
    }
    vectors->hessenberg_rhs[0] = scalars->beta;

    auto v = vectors->v_basis;
    blas::copy(global_length, vectors->residual, vectors->inc, v, vectors->inc,
               queue);

    blas::scale(global_length, 1.0 / scalars->beta, v, vectors->inc, queue);
}

template <typename value_type_in, typename value_type, typename index_type, ContextType device_type>
void step_1(dim2 size, matrix::Dense<value_type, device_type>* mtx,
            preconditioner::preconditioner<value_type_in, value_type,
                                           index_type, device_type>* precond,
            temp_scalars<value_type, index_type>* scalars,
            temp_vectors<value_type_in, value_type, index_type, device_type>* vectors,
            index_type cur_iter, magma_queue_t queue)
{
    auto global_len = size[0] + size[1];
    auto z = vectors->z_basis + global_len * cur_iter;
    auto v = vectors->v_basis + global_len * cur_iter;
    auto w = vectors->v_basis + global_len * (cur_iter + 1);

    blas::copy(global_len, v, 1, z, 1, queue);
    precond->apply(MagmaNoTrans, z, 1);

    fgmres::gemv(MagmaNoTrans, size[0], size[1], 1.0, mtx->get_values(),
                 size[0], z, 1, 0.0, w, 1, vectors->temp, queue);
    precond->apply(MagmaTrans, w, 1);
}

template <typename value_type_in, typename value_type, typename index_type, ContextType device_type>
void step_2(dim2 size, matrix::Dense<value_type, device_type>* mtx,
            matrix::Dense<value_type, device_type>* sol,
            preconditioner::preconditioner<value_type_in, value_type,
                                           index_type, device_type>* precond,
            temp_scalars<value_type, index_type>* scalars,
            temp_vectors<value_type_in, value_type, index_type, device_type>* vectors,
            index_type cur_iter, index_type max_iter, magma_queue_t queue)
{
    magma_queue_sync(queue);
    auto global_len = size[0] + size[1];
    auto ld = max_iter + 1;
    auto hessenberg_mtx = vectors->hessenberg_mtx;
    dim2 size_h = {max_iter + 1, max_iter};
    auto w = vectors->v_basis + global_len * (cur_iter + 1);
    for (index_type i = 0; i < cur_iter + 1; i++) {
        auto v = vectors->v_basis + global_len*i;
        hessenberg_mtx[i + ld*cur_iter] =
            blas::dot(global_len, v, 1, w, 1, queue);

        blas::axpy(global_len, -hessenberg_mtx[i + ld * cur_iter], v,
                   1, w, 1, queue);
    }
    auto w_norm = blas::norm2(global_len, w, 1, queue);
    magma_queue_sync(queue);
    hessenberg_mtx[cur_iter + 1 + ld*cur_iter] = w_norm;
    scalars->h = w_norm;


    // Solve here using givens qr.
    fgmres::givens_qr(size, hessenberg_mtx,
              (max_iter + 1), vectors->hessenberg_rhs,
              cur_iter, max_iter, *vectors->givens_cache.get());
    magma_queue_sync(queue);


    auto hessenberg_mtx_gpu = vectors->hessenberg_mtx_gpu;
    auto hessenberg_rhs_gpu = vectors->hessenberg_rhs_gpu;
    memory::setmatrix(max_iter + 1, max_iter, hessenberg_mtx, max_iter + 1,
              hessenberg_mtx_gpu, max_iter + 1, queue);
    memory::setmatrix(max_iter + 1, 1, vectors->hessenberg_rhs, max_iter + 1,
              hessenberg_rhs_gpu, max_iter + 1, queue);

    blas::trsv(MagmaUpper, MagmaNoTrans, MagmaNonUnit,
               cur_iter + 1, hessenberg_mtx_gpu, (max_iter + 1),
               hessenberg_rhs_gpu, 1, queue);

    magma_queue_sync(queue);
    blas::gemv(MagmaNoTrans, global_len, cur_iter + 1, 1.0, vectors->z_basis,
               global_len, hessenberg_rhs_gpu, 1,
               0.0, sol->get_values(), 1, queue);
}

template <typename value_type, typename index_type, ContextType device_type>
bool check_stopping_criteria(matrix::Dense<value_type, device_type>* mtx_in,
                             matrix::Dense<value_type, device_type>* rhs_in,
                             matrix::Dense<value_type, device_type>* sol_in,
                             value_type* res_vector, index_type max_iter,
                             double tolerance, index_type* iter,
                             double* resnorm, magma_queue_t queue)
{
    auto num_rows = mtx_in->get_size()[0];
    auto num_cols = mtx_in->get_size()[1];
    auto mtx = mtx_in->get_values();
    auto rhs = rhs_in->get_values();
    auto sol = sol_in->get_values() + num_rows;
    auto global_length = num_rows + num_cols;
    *iter += 1;
    index_type inc = 1;
    blas::copy(num_rows, rhs, inc, res_vector, inc, queue);
    blas::gemv(MagmaNoTrans, num_rows, num_cols, -1.0, mtx, num_rows, sol, inc,
               1.0, res_vector, inc, queue);
    auto rhsnorm = blas::norm2(num_rows, rhs, inc, queue);
    *resnorm = blas::norm2(num_rows, res_vector, inc, queue);
    *resnorm = *resnorm / rhsnorm;
    if ((*iter >= max_iter) || (*resnorm < tolerance)) {
        return true;
    } else {
        return false;
    }
}


}  // namespace fgmres

template <typename value_type_in, typename value_type, typename index_type, ContextType device_type=CUDA>
void run_fgmres(
    matrix::Dense<value_type, device_type>* mtx,
    matrix::Dense<value_type, device_type>* rhs,
    matrix::Dense<value_type, device_type>* sol,
    preconditioner::preconditioner<value_type_in, value_type, index_type, device_type>*
        precond,
    fgmres::temp_scalars<value_type, index_type>* scalars,
    fgmres::temp_vectors<value_type_in, value_type, index_type, device_type>* vectors,
    magma_int_t max_iter, double tolerance, magma_int_t* iter, double* resnorm,
    magma_queue_t queue, double* t_solve)
{
    *iter = 0;
    *t_solve = 0.0;
    auto size = mtx->get_size();
    auto global_length = size[0] + size[1];
    fgmres::initialize<double, double, magma_int_t>(size, mtx->get_values(),
        sol->get_values(), rhs->get_values(), precond, scalars, vectors, queue);
    double t = magma_sync_wtime(queue);
    while (1) {
        fgmres::step_1(size, mtx, precond, scalars, vectors, *iter, queue);
        fgmres::step_2(size, mtx, sol, precond, scalars, vectors, *iter,
                       max_iter, queue);
        if (fgmres::check_stopping_criteria(mtx, rhs, sol, vectors->residual,
                                          max_iter, tolerance, iter, resnorm,
                                          queue))
        {
            break;
        }
        blas::scale(global_length, 1.0 / scalars->h,
            &vectors->v_basis[global_length * (*iter)], vectors->inc, queue);
    }
    *t_solve += (magma_sync_wtime(queue) - t);
    std::cout << "*iter: " << *iter << ", max_iter: " <<  max_iter << ", *resnorm: " << *resnorm << ", t_solve (sec): " << *t_solve << "\n\n";
}


template void run_fgmres(
    matrix::Dense<double, CUDA>* mtx, matrix::Dense<double, CUDA>* rhs,
    matrix::Dense<double, CUDA>* sol,
    preconditioner::preconditioner<double, double, magma_int_t, CUDA>*
        precond,
    fgmres::temp_scalars<double, magma_int_t>* scalars,
    fgmres::temp_vectors<double, double, magma_int_t, CUDA>* vectors,
    magma_int_t max_iter, double tolerance, magma_int_t* iter, double* resnorm,
    magma_queue_t queue, double* t_solve);


}  // namespace solver
}  // namespace rls
