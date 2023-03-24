#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../blas/blas.hpp"
#include "../dense/dense.hpp"
#include "../memory/memory.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "base_types.hpp"
#include "lsqr.hpp"
#include "solver.hpp"

#include "../../utils/io.hpp"
#include "../../cuda/solver/lsqr_kernels.cuh"


namespace rls {
namespace solver {
namespace {

void getmatrix(magma_int_t num_rows, magma_int_t num_cols,
               double* source_vector, magma_int_t ld_source,
               double* dest_vector, magma_int_t ld_dest, magma_queue_t queue)
{
    magma_dgetmatrix(num_rows, num_cols, source_vector, ld_source, dest_vector,
                     ld_source, queue);
}

void getmatrix(magma_int_t num_rows, magma_int_t num_cols, float* source_vector,
               magma_int_t ld_source, float* dest_vector, magma_int_t ld_dest,
               magma_queue_t queue)
{
    magma_sgetmatrix(num_rows, num_cols, source_vector, ld_source, dest_vector,
                     ld_source, queue);
}

void precond_apply(magma_trans_t trans, magma_int_t num_rows,
                   double* precond_mtx, magma_int_t ld, double* u_vector,
                   magma_int_t inc_u, magma_queue_t queue)
{
    std::cout << "num_rows: " << num_rows << ", ld: " << ld << ", inc_u: " << inc_u << '\n';
    cudaDeviceSynchronize();
    blas::trsv(MagmaUpper, trans, MagmaNonUnit, num_rows, precond_mtx, ld,
               u_vector, inc_u, queue);
}

void precond_apply(magma_trans_t trans, magma_int_t num_rows,
                   float* precond_mtx, magma_int_t ld, float* u_vector,
                   magma_int_t inc_u, magma_queue_t queue)
{
    blas::trsv(MagmaUpper, trans, MagmaNonUnit, num_rows, precond_mtx, ld,
               u_vector, inc_u, queue);
}


template <typename value_type, typename index_type>
void initialize(index_type num_rows, index_type num_cols, index_type* iter,
                value_type* alpha, value_type* beta, value_type* rho_bar,
                value_type* phi_bar, value_type* mtx, value_type* u_vector,
                value_type* v_vector, value_type* w_vector, value_type* rhs,
                magma_queue_t queue)
{
    index_type inc = 1;
    *iter = 0;
    *beta = blas::norm2(num_rows, rhs, inc, queue);
    blas::copy(num_rows, rhs, inc, u_vector, inc, queue);
    blas::scale(num_rows, 1 / *beta, u_vector, inc, queue);
    blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, u_vector,
               inc, 1.0, v_vector, inc, queue);


    *alpha = blas::norm2(num_rows, v_vector, inc, queue);
    blas::scale(num_rows, 1 / *alpha, v_vector, inc, queue);
    blas::copy(num_rows, v_vector, inc, w_vector, inc, queue);
    *rho_bar = *alpha;
    *phi_bar = *beta;
}

template <typename value_type, typename index_type>
void initialize(index_type num_rows, index_type num_cols, index_type* iter,
                value_type* alpha, value_type* beta, value_type* rho_bar,
                value_type* phi_bar, value_type* mtx, value_type* u_vector,
                value_type* v_vector, value_type* w_vector, value_type* rhs,
                value_type* precond_mtx, index_type ld_precond,
                magma_queue_t queue)
{
    index_type inc = 1;
    *iter = 0;
    *beta = blas::norm2(num_rows, rhs, inc, queue);
    blas::copy(num_rows, rhs, inc, u_vector, inc, queue);
    blas::scale(num_rows, 1 / *beta, u_vector, inc, queue);

    blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, u_vector,
               inc, 0.0, v_vector, inc, queue);

    precond_apply(MagmaTrans, num_cols, precond_mtx, ld_precond, v_vector, inc,
                  queue);
    *alpha = blas::norm2(num_cols, v_vector, inc, queue);
    blas::scale(num_cols, 1 / *alpha, v_vector, inc, queue);
    blas::copy(num_cols, v_vector, inc, w_vector, inc, queue);
    *phi_bar = *beta;
    *rho_bar = *alpha;
}

template <typename value_type_in, typename value_type, typename index_type>
void initialize(index_type num_rows, index_type num_cols, value_type* mtx,
                value_type* rhs, index_type* iter,
                temp_scalars<value_type, index_type>* scalars,
                temp_vectors<value_type_in, value_type, index_type>* vectors,
                magma_queue_t queue)
{
    vectors->inc = 1;
    memory::malloc(&vectors->u, num_rows);
    memory::malloc(&vectors->v, num_rows);
    memory::malloc(&vectors->w, num_rows);
    memory::malloc(&vectors->temp, num_rows);
    if (!std::is_same<value_type_in, value_type>::value) {
        memory::malloc(&vectors->u_in, num_rows);
        memory::malloc(&vectors->v_in, num_rows);
        memory::malloc(&vectors->temp_in, num_rows);
        memory::malloc(&vectors->mtx_in, num_rows * num_cols);

        if (sizeof(value_type) > sizeof(value_type_in)) {
            cuda::demote(num_rows, num_cols, mtx, num_rows, vectors->mtx_in,
                         num_rows);
        } else if (sizeof(value_type) < sizeof(value_type_in)) {
            cuda::promote(num_rows, num_cols, mtx, num_rows, vectors->mtx_in,
                          num_rows);
        }
    }

    *iter = 0;
    scalars->beta = blas::norm2(num_rows, rhs, vectors->inc, queue);
    blas::copy(num_rows, rhs, vectors->inc, vectors->u, vectors->inc, queue);
    blas::scale(num_rows, 1 / scalars->beta, vectors->u, vectors->inc, queue);
    blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, vectors->u,
               vectors->inc, 1.0, vectors->v, vectors->inc, queue);

    scalars->alpha = blas::norm2(num_cols, vectors->v, vectors->inc, queue);
    blas::scale(num_cols, 1 / scalars->alpha, vectors->v, vectors->inc, queue);
    blas::copy(num_cols, vectors->v, vectors->inc, vectors->w, vectors->inc,
               queue);
    scalars->rho_bar = scalars->alpha;
    scalars->phi_bar = scalars->beta;
}


// template <typename value_type_in, typename value_type, typename index_type>
// void initialize(dim2 size, value_type* mtx, value_type* rhs,
//                 value_type* precond_mtx, index_type ld_precond,
//                 temp_scalars<value_type, index_type>* scalars,
//                 temp_vectors<value_type_in, value_type, index_type>* vectors,
//                 magma_queue_t queue, double* t_solve)
// {
//     index_type num_rows = size[0];
//     index_type num_cols = size[1];
//     vectors->inc = 1;
//     memory::malloc(&vectors->u, num_rows);
//     memory::malloc(&vectors->v, num_rows);
//     memory::malloc(&vectors->w, num_rows);
//     memory::malloc(&vectors->temp, num_rows);
//     if (!std::is_same<value_type_in, value_type>::value) {
//         memory::malloc(&vectors->u_in, num_rows);
//         memory::malloc(&vectors->v_in, num_rows);
//         memory::malloc(&vectors->temp_in, num_rows);
//         memory::malloc(&vectors->mtx_in, num_rows * num_cols);

//         // if (sizeof(value_type) > sizeof(value_type_in)) {
//             cuda::demote(num_rows, num_cols, mtx, num_rows, vectors->mtx_in,
//             num_rows);
//         // }
//         // else if (sizeof(value_type) < sizeof(value_type_in)) {
//         //     cuda::promote(num_rows, num_cols, mtx, num_rows,
//         vectors->mtx_in, num_rows);
//         // }
//     }

//     double t = magma_sync_wtime(queue);
//     scalars->beta = blas::norm2(num_rows, rhs, vectors->inc, queue);
//     blas::copy(num_rows, rhs, vectors->inc, vectors->u, vectors->inc, queue);
//     blas::scale(num_rows, 1 / scalars->beta, vectors->u, vectors->inc,
//     queue);

//     if (!std::is_same<value_type_in, value_type>::value) {
//         cuda::demote(num_rows, 1, vectors->u, num_rows,
//                      vectors->u_in, num_rows);
//         cuda::demote(num_rows, 1, vectors->v, num_rows, vectors->v_in,
//                      num_rows);
//         blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
//         num_rows, vectors->u_in,
//                vectors->inc, 0.0, vectors->v_in, vectors->inc, queue);
//         cuda::promote(num_rows, 1, vectors->v_in,
//             num_rows, vectors->v, num_rows);
//     }
//     else {
//         blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows,
//         vectors->u,
//                vectors->inc, 0.0, vectors->v, vectors->inc, queue);
//     }

//     precond_apply(MagmaTrans, num_cols, precond_mtx, ld_precond, vectors->v,
//                   vectors->inc, queue);
//     scalars->alpha = blas::norm2(num_cols, vectors->v, vectors->inc, queue);
//     blas::scale(num_cols, 1 / scalars->alpha, vectors->v, vectors->inc,
//     queue); blas::copy(num_cols, vectors->v, vectors->inc, vectors->w,
//     vectors->inc, queue); scalars->phi_bar = scalars->beta; scalars->rho_bar
//     = scalars->alpha; *t_solve += (magma_sync_wtime(queue) - t);
// }

template <typename value_type_in, typename value_type, typename index_type>
void initialize(dim2 size, value_type* mtx, value_type* rhs,
                preconditioner::preconditioner<value_type_in, value_type,
                                               index_type>* precond,
                temp_scalars<value_type, index_type>* scalars,
                temp_vectors<value_type_in, value_type, index_type>* vectors,
                magma_queue_t queue, double* t_solve)
{
    // auto precond_mtx = precond->get_mtx();
    // index_type ld_precond = precond_mtx->get_size()[0];
    std::cout << "in initialize\n";
    index_type num_rows = size[0];
    index_type num_cols = size[1];
    vectors->inc = 1;
    std::cout << "num_rows: " << num_rows << '\n';
    if (!std::is_same<value_type_in, value_type>::value) {
        // if (sizeof(value_type) > sizeof(value_type_in)) {
        cuda::demote(num_rows, num_cols, mtx, num_rows, vectors->mtx_in,
                     num_rows);
        // }
        // else if (sizeof(value_type) < sizeof(value_type_in)) {
        //     cuda::promote(num_rows, num_cols, mtx, num_rows, vectors->mtx_in,
        //     num_rows);
        // }
    }

    // double t = magma_sync_wtime(queue);
    scalars->beta = blas::norm2(num_rows, rhs, vectors->inc, queue);
    std::cout << "scalars->beta: " << scalars->beta << '\n';
    std::cout << "vectors->inc: " << vectors->inc << '\n';
    blas::copy(num_rows, rhs, vectors->inc, vectors->u, vectors->inc, queue);
    magma_queue_sync(queue);
    blas::scale(num_rows, 1 / scalars->beta, vectors->u, vectors->inc,
        queue);
    magma_queue_sync(queue);

    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors->u, num_rows,
                     vectors->u_in, num_rows);
        cuda::demote(num_rows, 1, vectors->v, num_rows, vectors->v_in,
                     num_rows);
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
            num_rows, vectors->u_in, vectors->inc, 0.0, vectors->v_in,
            vectors->inc, queue);
        magma_queue_sync(queue);
        cuda::promote(num_rows, 1, vectors->v_in,
            num_rows, vectors->v, num_rows);
    }
    else {
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, vectors->u,
               vectors->inc, 0.0, vectors->v, vectors->inc, queue);
        magma_queue_sync(queue);
    }
    std::cout << "v[0]\n";
    rls::io::print_mtx_gpu(10, 1, (double*)vectors->v, num_cols, queue);
    // value_type one = 0.0;
    // cuda::set_values(num_cols, one, vectors->v);
    std::cout << "before apply\n";
    cudaDeviceSynchronize();
    auto P = precond->get_values();
magma_int_t ld_precond = 1500;
    io::write_mtx("precond_mtx2.mtx", ld_precond, num_cols, (double*)precond->get_values(), ld_precond, queue);
    io::write_mtx("v2.mtx", num_cols, 1, (double*)vectors->v, num_cols, queue); 
    // precond_apply(MagmaTrans, num_cols, P, 1500, vectors->v, 1, queue);
    magma_queue_sync(queue);
    precond->apply(MagmaTrans, vectors->v, vectors->inc);
    cudaDeviceSynchronize();


    std::cout << "(after) v[0]\n";
    rls::io::print_mtx_gpu(10, 1, (double*)vectors->v, num_cols, queue);

    scalars->alpha = blas::norm2(num_cols, vectors->v, vectors->inc, queue);
    magma_queue_sync(queue);
    std::cout << "scalars->alpha: " << scalars->alpha << '\n';
    blas::scale(num_cols, 1 / scalars->alpha, vectors->v, vectors->inc,
        queue);
    magma_queue_sync(queue);
    blas::copy(num_cols, vectors->v, vectors->inc, vectors->w,
        vectors->inc, queue);
    magma_queue_sync(queue);
    scalars->phi_bar = scalars->beta;
    scalars->rho_bar = scalars->alpha;
    // *t_solve += (magma_sync_wtime(queue) - t);
}


template <typename value_type_in, typename value_type, typename index_type>
void finalize(temp_vectors<value_type_in, value_type, index_type>* vectors)
{
    memory::free(vectors->u);
    memory::free(vectors->v);
    memory::free(vectors->w);
    memory::free(vectors->temp);
    if (!std::is_same<value_type_in, value_type>::value) {
        memory::free(vectors->u_in);
        memory::free(vectors->v_in);
        memory::free(vectors->mtx_in);
        memory::free(vectors->temp_in);
    }
}


template <typename value_type, typename index_type>
void step_1(index_type num_rows, index_type num_cols, value_type* alpha,
            value_type* beta, value_type* mtx, value_type* u_vector,
            value_type* v_vector, magma_queue_t queue)
{
    index_type inc = 1;
    blas::scale(num_rows, *alpha, u_vector, inc, queue);
    blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows, v_vector,
               inc, -1.0, u_vector, inc, queue);
    *beta = blas::norm2(num_rows, u_vector, inc, queue);
    blas::scale(num_rows, 1 / *beta, u_vector, inc, queue);
    blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, u_vector,
               inc, -(*beta), v_vector, inc, queue);
    *alpha = blas::norm2(num_cols, v_vector, inc, queue);
    blas::scale(num_rows, 1 / *alpha, v_vector, inc, queue);
}

template <typename value_type, typename index_type>
void step_1(index_type num_rows, index_type num_cols, value_type* alpha,
            value_type* beta, value_type* mtx, value_type* u_vector,
            value_type* v_vector, value_type* precond_mtx,
            index_type ld_precond, value_type* tmp_vector, magma_queue_t queue)
{
    index_type inc = 1;
    // compute new u_vector
    blas::scale(num_rows, *alpha, u_vector, inc, queue);
    blas::copy(num_cols, v_vector, inc, tmp_vector, inc, queue);
    precond_apply(MagmaNoTrans, num_cols, precond_mtx, ld_precond, tmp_vector,
                  inc, queue);
    blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows, tmp_vector,
               inc, -1.0, u_vector, inc, queue);
    *beta = blas::norm2(num_rows, u_vector, inc, queue);
    blas::scale(num_rows, 1 / *beta, u_vector, inc, queue);

    // compute new v_vector
    blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, u_vector,
               inc, 0.0, tmp_vector, inc, queue);
    precond_apply(MagmaTrans, num_cols, precond_mtx, ld_precond, tmp_vector,
                  inc, queue);
    blas::axpy(num_cols, -(*beta), v_vector, 1, tmp_vector, 1, queue);
    *alpha = blas::norm2(num_cols, tmp_vector, inc, queue);
    blas::scale(num_cols, 1 / *alpha, tmp_vector, inc, queue);
    blas::copy(num_cols, tmp_vector, inc, v_vector, inc, queue);
}

template <typename value_type_in, typename value_type, typename index_type>
void step_1(index_type num_rows, index_type num_cols, value_type* mtx,
            value_type* precond_mtx, index_type ld_precond,
            temp_scalars<value_type, index_type>* scalars,
            temp_vectors<value_type_in, value_type, index_type>* vectors,
            magma_queue_t queue)
{
    index_type inc = 1;
    // compute new u_vector
    blas::scale(num_rows, scalars->alpha, vectors->u, inc, queue);
    blas::copy(num_cols, vectors->v, inc, vectors->temp, inc, queue);
    precond_apply(MagmaNoTrans, num_cols, precond_mtx, ld_precond,
                  vectors->temp, inc, queue);
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors->temp, num_rows, vectors->temp_in,
                     num_rows);
        cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in,
                     num_rows);
        cudaDeviceSynchronize();
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
                   num_rows, vectors->temp_in, inc, -1.0, vectors->u_in, inc,
                   queue);
        cuda::promote(num_rows, 1, vectors->u_in, num_rows, vectors->u,
                      num_rows);
        cudaDeviceSynchronize();
    } else {
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows,
                   vectors->temp, inc, -1.0, vectors->u, inc, queue);
    }
    scalars->beta = blas::norm2(num_rows, vectors->u, inc, queue);
    blas::scale(num_rows, 1 / scalars->beta, vectors->u, inc, queue);

    // compute new v_vector
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in,
                     num_rows);
        cuda::demote(num_rows, 1, vectors->temp, num_rows, vectors->temp_in,
                     num_rows);
        cudaDeviceSynchronize();
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
                   num_rows, vectors->u_in, inc, 0.0, vectors->temp_in, inc,
                   queue);
        cuda::promote(num_rows, 1, vectors->temp_in, num_rows, vectors->temp,
                      num_rows);
        cudaDeviceSynchronize();
    } else {
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows,
                   vectors->u, inc, 0.0, vectors->temp, inc, queue);
    }

    precond_apply(MagmaTrans, num_cols, precond_mtx, ld_precond, vectors->temp,
                  inc, queue);
    blas::axpy(num_cols, -(scalars->beta), vectors->v, 1, vectors->temp, 1,
               queue);
    scalars->alpha = blas::norm2(num_cols, vectors->temp, inc, queue);

    blas::scale(num_cols, 1 / scalars->alpha, vectors->temp, inc, queue);
    blas::copy(num_cols, vectors->temp, inc, vectors->v, inc, queue);
}

template <typename value_type, typename index_type>
void step_2(index_type num_rows, index_type num_cols, value_type alpha,
            value_type beta, value_type* mtx, value_type* rho_bar,
            value_type* phi_bar, value_type* u_vector, value_type* v_vector,
            value_type* w_vector, value_type* rhs, value_type* sol,
            magma_queue_t queue)
{
    index_type inc = 1;
    auto rho = std::sqrt(((*rho_bar) * (*rho_bar) + beta * beta));
    auto c = (*rho_bar) / rho;
    auto s = beta / rho;
    auto theta = s * alpha;
    *rho_bar = -c * alpha;
    auto phi = c * (*phi_bar);
    (*phi_bar) = s * (*phi_bar);
    blas::axpy(num_cols, phi / rho, w_vector, 1, sol, 1, queue);
    blas::scale(num_cols, -(theta / rho), w_vector, inc, queue);
    blas::axpy(num_cols, 1.0, v_vector, 1, w_vector, 1, queue);
}

template <typename value_type, typename index_type>
void step_2(index_type num_rows, index_type num_cols, value_type alpha,
            value_type beta, value_type* mtx, value_type* rho_bar,
            value_type* phi_bar, value_type* u_vector, value_type* v_vector,
            value_type* w_vector, value_type* rhs, value_type* sol,
            value_type* precond_mtx, index_type ld_precond,
            value_type* tmp_vector, magma_queue_t queue)
{
    index_type inc = 1;
    auto rho = std::sqrt(((*rho_bar) * (*rho_bar) + beta * beta));
    auto c = (*rho_bar) / rho;
    auto s = beta / rho;
    auto theta = s * alpha;
    *rho_bar = -c * alpha;
    auto phi = c * (*phi_bar);
    (*phi_bar) = s * (*phi_bar);
    blas::copy(num_cols, w_vector, inc, tmp_vector, inc, queue);
    precond_apply(MagmaNoTrans, num_cols, precond_mtx, ld_precond, tmp_vector,
                  inc, queue);
    blas::axpy(num_cols, phi / rho, tmp_vector, 1, sol, 1, queue);
    // compute new w_vector
    blas::scale(num_cols, -(theta / rho), w_vector, inc, queue);
    blas::axpy(num_cols, 1.0, v_vector, 1, w_vector, 1, queue);
}

template <typename value_type_in, typename value_type, typename index_type>
void step_2(index_type num_rows, index_type num_cols, value_type* mtx,
            value_type* rhs, value_type* sol, value_type* precond_mtx,
            index_type ld_precond,
            temp_scalars<value_type, index_type>* scalars,
            temp_vectors<value_type_in, value_type, index_type>* vectors,
            magma_queue_t queue)
{
    index_type inc = 1;
    auto rho = std::sqrt(((scalars->rho_bar) * (scalars->rho_bar) +
                          scalars->beta * scalars->beta));
    auto c = (scalars->rho_bar) / rho;
    auto s = scalars->beta / rho;
    auto theta = s * scalars->alpha;
    scalars->rho_bar = -c * scalars->alpha;
    auto phi = c * (scalars->phi_bar);
    scalars->phi_bar = s * (scalars->phi_bar);
    blas::copy(num_cols, vectors->w, inc, vectors->temp, inc, queue);
    precond_apply(MagmaNoTrans, num_cols, precond_mtx, ld_precond,
                  vectors->temp, inc, queue);
    blas::axpy(num_cols, phi / rho, vectors->temp, 1, sol, 1, queue);
    // compute new w_vector
    blas::scale(num_cols, -(theta / rho), vectors->w, inc, queue);
    blas::axpy(num_cols, 1.0, vectors->v, 1, vectors->w, 1, queue);
}


template <typename value_type, typename index_type>
void allocate_memory(index_type num_rows, index_type num_cols,
                     value_type** u_vector, value_type** v_vector,
                     value_type** w_vector, value_type** tmp_vector)
{
    memory::malloc(u_vector, num_rows);
    memory::malloc(v_vector, num_rows);
    memory::malloc(w_vector, num_cols);
    memory::malloc(tmp_vector, num_rows);
}

template <typename value_type>
void free_memory(value_type* u_vector, value_type* v_vector,
                 value_type* w_vector, value_type* tmp_vector)
{
    magma_free(u_vector);
    magma_free(v_vector);
    magma_free(w_vector);
    magma_free(tmp_vector);
}

}  // end of anonymous namespace


template <typename value_type_in, typename value_type, typename index_type>
void lsqr<value_type_in, value_type, index_type>::step_1(
    matrix::dense<value_type> mtx,
    temp_scalars<value_type, index_type>* scalars,
    temp_vectors<value_type_in, value_type, index_type>* vectors)
{
    auto queue = context->get_queue();
    index_type inc = 1;
    auto size = mtx.get_size();
    auto num_rows = size[0];
    auto num_cols = size[1];
    // compute new u_vector
    blas::scale(num_rows, scalars->alpha, vectors->u, inc, queue);
    blas::copy(num_cols, vectors->v, inc, vectors->temp, inc, queue);
    // precond.apply(MagmaNoTrans, vectors->temp, inc);
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors->temp, num_rows, vectors->temp_in,
                     num_rows);
        cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in,
                     num_rows);
        cudaDeviceSynchronize();
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
                   num_rows, vectors->temp_in, inc, -1.0, vectors->u_in, inc,
                   queue);
        cuda::promote(num_rows, 1, vectors->u_in, num_rows, vectors->u,
                      num_rows);
        cudaDeviceSynchronize();
    } else {
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows,
                   vectors->temp, inc, -1.0, vectors->u, inc, queue);
    }
    scalars->beta = blas::norm2(num_rows, vectors->u, inc, queue);
    blas::scale(num_rows, 1 / scalars->beta, vectors->u, inc, queue);

    // compute new v_vector
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in,
                     num_rows);
        cuda::demote(num_rows, 1, vectors->temp, num_rows, vectors->temp_in,
                     num_rows);
        cudaDeviceSynchronize();
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
                   num_rows, vectors->u_in, inc, 0.0, vectors->temp_in, inc,
                   queue);
        cuda::promote(num_rows, 1, vectors->temp_in, num_rows, vectors->temp,
                      num_rows);
        cudaDeviceSynchronize();
    } else {
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows,
                   vectors->u, inc, 0.0, vectors->temp, inc, queue);
    }

    // precond.apply(MagmaTrans, vectors->temp, inc);
    blas::axpy(num_cols, -(scalars->beta), vectors->v, 1, vectors->temp, 1,
               queue);
    scalars->alpha = blas::norm2(num_cols, vectors->temp, inc, queue);

    blas::scale(num_cols, 1 / scalars->alpha, vectors->temp, inc, queue);
    blas::copy(num_cols, vectors->temp, inc, vectors->v, inc, queue);
}

// template <typename value_type_in, typename value_type, typename index_type>
// void step_1(matrix::dense<value_type>* mtx_in,
//             preconditioner::preconditioner<value_type_in, value_type,
//                                            index_type>* precond,
//             temp_scalars<value_type, index_type>* scalars,
//             temp_vectors<value_type_in, value_type, index_type>* vectors,
//             magma_queue_t queue)
// {
//     std::cout << "inside step1" << '\n';
//     index_type inc = 1;
//     auto size = mtx_in->get_size();
//     auto mtx = mtx_in->get_values();
//     auto num_rows = size[0];
//     auto num_cols = size[1];
//     // compute new u_vector
//     blas::scale(num_rows, scalars->alpha, vectors->u, inc, queue);
//     blas::copy(num_cols, vectors->v, inc, vectors->temp, inc, queue);
//     cudaDeviceSynchronize();

//     std::cout << "temp (before)\n";
//     rls::io::print_mtx_gpu(10, 1, (double*)vectors->temp, num_cols, queue);

//     auto P = precond->get_values();
//     rls::io::print_mtx_gpu(10, 10, (double*)P, 1500, queue);
//     std::cout << "num_cols: " << num_cols << '\n';
//     cudaDeviceSynchronize();
//     // precond_apply(MagmaNoTrans, num_cols, P, 1500, vectors->temp, num_cols, queue);
//     precond->apply(MagmaNoTrans, vectors->temp, inc);

//     std::cout << "temp (after)\n";
//     cudaDeviceSynchronize();
//     rls::io::print_mtx_gpu(10, 1, (double*)vectors->temp, num_cols, queue);

//     if (!std::is_same<value_type_in, value_type>::value) {
//         cuda::demote(num_rows, 1, vectors->temp, num_rows, vectors->temp_in,
//             num_rows);
//         cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in, num_rows);
//         cudaDeviceSynchronize();
//         blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
//             num_rows, vectors->temp_in, inc, -1.0, vectors->u_in, inc, queue);
//         cuda::promote(num_rows, 1, vectors->u_in, num_rows, vectors->u,
//             num_rows);
//         cudaDeviceSynchronize();
//     }
//     else {
//         std::cout << "u (before)\n";
//         rls::io::print_mtx_gpu(10, 1, (double*)vectors->u, num_rows, queue);
//         blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows,
//             vectors->temp, inc, -1.0, vectors->u, inc, queue);
//     }
//     scalars->beta = blas::norm2(num_rows, vectors->u, inc, queue);
//     std::cout << "beta-after-step1: " << scalars->beta << '\n';
//     std::cout << "u (after)\n";
//     rls::io::print_mtx_gpu(10, 1, (double*)vectors->u, num_rows, queue);
//     blas::scale(num_rows, 1.0 / scalars->beta, vectors->u, inc, queue);

//     // compute new v_vector
//     if (!std::is_same<value_type_in, value_type>::value) {
//         cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in,
//                      num_rows);
//         cuda::demote(num_rows, 1, vectors->temp, num_rows,
//                      vectors->temp_in, num_rows);
//         cudaDeviceSynchronize();
//         blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
//         num_rows, vectors->u_in,
//                inc, 0.0, vectors->temp_in, inc, queue);
//         cuda::promote(num_rows, 1, vectors->temp_in,
//             num_rows, vectors->temp, num_rows);
//         cudaDeviceSynchronize();
//     }
//     else {
//         std::cout << "temp (before)\n";
//         rls::io::print_mtx_gpu(10, 1, (double*)vectors->temp, num_cols, queue);
//         std::cout << "inc: " << inc << '\n';
//         blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows,
//             vectors->u, inc, 0.0, vectors->temp, inc, queue);
//         std::cout << "temp (after)\n";
//         rls::io::print_mtx_gpu(10, 1, (double*)vectors->temp, num_cols, queue);
//     }

//     std::cout << "temp (before)\n";
//     cudaDeviceSynchronize();
//     rls::io::print_mtx_gpu(10, 1, (double*)vectors->temp, num_cols, queue);
//     cudaDeviceSynchronize();
//     precond->apply(MagmaNoTrans, vectors->temp, inc);
//     cudaDeviceSynchronize();
//     precond_apply(MagmaTrans, num_cols, P, 1500, vectors->temp, num_cols, queue);
//     cudaDeviceSynchronize();
//     std::cout << "temp (after)\n";
//     cudaDeviceSynchronize();
//     rls::io::print_mtx_gpu(10, 1, (double*)vectors->temp, num_cols, queue);
//     blas::axpy(num_cols, -(scalars->beta), vectors->v, 1, vectors->temp, 1,
//         queue);
//     cudaDeviceSynchronize();
//     scalars->alpha = blas::norm2(num_cols, vectors->temp, inc, queue);
//     std::cout << "alpha-after-step1: " << scalars->alpha << '\n';

//     blas::scale(num_cols, 1 / scalars->alpha, vectors->temp, inc, queue);
//     blas::copy(num_cols, vectors->temp, inc, vectors->v, inc, queue);
// }


// Step 1 of preconditioned LSQR.
// template <typename value_type_in, typename value_type, typename index_type>
// void step_1(index_type num_rows, index_type num_cols, value_type* mtx,
            // value_type* precond_mtx, index_type ld_precond,
            // temp_scalars<value_type, index_type>* scalars,
            // temp_vectors<value_type_in, value_type, index_type>* vectors,
            // magma_queue_t queue)
template <typename value_type_in, typename value_type, typename index_type>
void step_1(matrix::dense<value_type>* mtx_in,
            preconditioner::preconditioner<value_type_in, value_type,
                                           index_type>* precond,
            temp_scalars<value_type, index_type>* scalars,
            temp_vectors<value_type_in, value_type, index_type>* vectors,
            magma_queue_t queue)
{
    auto num_rows = mtx_in->get_size()[0];
    auto num_cols = mtx_in->get_size()[1];
    index_type ld_precond = precond->get_size()[0];
    auto precond_mtx = precond->get_values();
    auto mtx = mtx_in->get_values();

    index_type inc = 1;
    // compute new u_vector
    blas::scale(num_rows, scalars->alpha, vectors->u, inc, queue);
    magma_queue_sync(queue);
    blas::copy(num_cols, vectors->v, inc, vectors->temp, inc, queue);
    precond->apply(MagmaNoTrans, vectors->temp, inc);
    magma_queue_sync(queue);
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors->temp, num_rows, vectors->temp_in,
                     num_rows);
        cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in, num_rows);
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
                   num_rows, vectors->temp_in, inc, -1.0, vectors->u_in, inc,
                   queue);
        magma_queue_sync(queue);
        cuda::promote(num_rows, 1, vectors->u_in, num_rows, vectors->u, num_rows);
    } else {
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows,
                   vectors->temp, inc, -1.0, vectors->u, inc, queue);
        magma_queue_sync(queue);
    }
    scalars->beta = blas::norm2(num_rows, vectors->u, inc, queue);
    blas::scale(num_rows, 1 / scalars->beta, vectors->u, inc, queue);
    magma_queue_sync(queue);

    // compute new v_vector
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors->u, num_rows, vectors->u_in, num_rows);
        cuda::demote(num_rows, 1, vectors->temp, num_rows, vectors->temp_in,
                     num_rows);
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors->mtx_in,
                   num_rows, vectors->u_in, inc, 0.0, vectors->temp_in, inc,
                   queue);
        magma_queue_sync(queue);
        cuda::promote(num_rows, 1, vectors->temp_in, num_rows, vectors->temp,
                      num_rows);
        cudaDeviceSynchronize();
    } else {
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows,
                   vectors->u, inc, 0.0, vectors->temp, inc, queue);
        magma_queue_sync(queue);
    }

    // precond->apply(MagmaTrans, vectors->temp, inc, queue);
    precond->apply(MagmaTrans, vectors->temp, inc);   // uses different queue apparently and causes problems
    blas::axpy(num_cols, -(scalars->beta), vectors->v, 1, vectors->temp, 1, queue);
    magma_queue_sync(queue);
    scalars->alpha = blas::norm2(num_cols, vectors->temp, inc, queue);
    magma_queue_sync(queue);

    blas::scale(num_cols, 1 / scalars->alpha, vectors->temp, inc, queue);
    magma_queue_sync(queue);
    blas::copy(num_cols, vectors->temp, inc, vectors->v, inc, queue);
    magma_queue_sync(queue);
}

template <typename value_type_in, typename value_type, typename index_type>
void lsqr<value_type_in, value_type, index_type>::step_2(
    matrix::dense<value_type>& mtx_in, matrix::dense<value_type>& sol_in,
    temp_scalars<value_type, index_type>* scalars,
    temp_vectors<value_type_in, value_type, index_type>* vectors)
{
    auto queue = context->get_queue();
    auto mtx = mtx_in.get_values();
    auto sol = sol_in.get_values();
    auto size_mtx = mtx.get_size();
    index_type inc = 1;
    auto rho = std::sqrt(((scalars->rho_bar) * (scalars->rho_bar) +
                          scalars->beta * scalars->beta));
    auto c = (scalars->rho_bar) / rho;
    auto s = scalars->beta / rho;
    auto theta = s * scalars->alpha;
    scalars->rho_bar = -c * scalars->alpha;
    auto phi = c * (scalars->phi_bar);
    scalars->phi_bar = s * (scalars->phi_bar);
    blas::copy(size_mtx[1], vectors->w, inc, vectors->temp, inc, queue);
    // precond.apply(MagmaNoTrans, vectors->temp, inc);
    blas::axpy(size_mtx[1], phi / rho, vectors->temp, 1, sol, 1, queue);
    // compute new w_vector
    blas::scale(size_mtx[1], -(theta / rho), vectors->w, inc, queue);
    blas::axpy(size_mtx[1], 1.0, vectors->v, 1, vectors->w, 1, queue);
}

// template <typename value_type_in, typename value_type, typename index_type>
// void step_2(matrix::dense<value_type>* mtx_in,
//             matrix::dense<value_type>* sol_in,
//             // matrix::dense<value_type>* precond,
//             preconditioner::preconditioner<value_type_in, value_type,
//                                            index_type>* precond,
//             temp_scalars<value_type, index_type>* scalars,
//             temp_vectors<value_type_in, value_type, index_type>* vectors,
//             magma_queue_t queue)
// {
//     std::cout << "in step_2\n";
//     auto mtx = mtx_in->get_values();
//     auto sol = sol_in->get_values();
//     auto size_mtx = mtx_in->get_size();
//     auto num_rows = size_mtx[0];
//     auto num_cols = size_mtx[1];
//     index_type inc = 1;
//     auto rho = std::sqrt(((scalars->rho_bar) * (scalars->rho_bar) +
//                           scalars->beta * scalars->beta));
//     auto c = (scalars->rho_bar) / rho;
//     auto s = scalars->beta / rho;
//     auto theta = s * scalars->alpha;
//     scalars->rho_bar = -c * scalars->alpha;
//     auto phi = c * (scalars->phi_bar);
//     scalars->phi_bar = s * (scalars->phi_bar);
//     blas::copy(size_mtx[1], vectors->w, inc, vectors->temp, inc, queue);
//     // precond->apply(MagmaNoTrans, size_precond[0], precond_mtx->get_values(),
//     // size_precond[0], vectors->temp, inc, queue); precond_apply(MagmaNoTrans,
//     // num_cols, precond_mtx, ld_precond, vectors->temp,
//     //               inc, queue);
//     precond->apply(MagmaNoTrans, vectors->temp, inc);
//     blas::axpy(size_mtx[1], phi / rho, vectors->temp, 1, sol, 1, queue);
//     // compute new w_vector
//     blas::scale(size_mtx[1], -(theta / rho), vectors->w, inc, queue);
//     blas::axpy(size_mtx[1], 1.0, vectors->v, 1, vectors->w, 1, queue);
// }

// Step 2 of preconditioned LSQR.
// template <typename value_type_in, typename value_type, typename index_type>
// void step_2(index_type num_rows, index_type num_cols, value_type* mtx,
            // value_type* rhs, value_type* sol, value_type* precond_mtx,
            // index_type ld_precond,
            // temp_scalars<value_type, index_type>& scalars,
            // temp_vectors<value_type_in, value_type, index_type>& vectors,
            // magma_queue_t queue)
template <typename value_type_in, typename value_type, typename index_type>
void step_2(matrix::dense<value_type>* mtx_in,
            matrix::dense<value_type>* sol_in,
            // matrix::dense<value_type>* precond,
            preconditioner::preconditioner<value_type_in, value_type,
                                           index_type>* precond,
            temp_scalars<value_type, index_type>* scalars,
            temp_vectors<value_type_in, value_type, index_type>* vectors,
            magma_queue_t queue)
{
    auto num_rows = mtx_in->get_size()[0];
    auto num_cols = mtx_in->get_size()[1];
    auto mtx = mtx_in->get_values();
    auto sol = sol_in->get_values();
    auto precond_mtx = precond->get_values();
    index_type ld_precond = precond->get_size()[0];
    index_type inc = 1;
    auto rho = std::sqrt(
        ((scalars->rho_bar) * (scalars->rho_bar) + scalars->beta * scalars->beta));
    auto c = (scalars->rho_bar) / rho;
    auto s = scalars->beta / rho;
    auto theta = s * scalars->alpha;
    scalars->rho_bar = -c * scalars->alpha;
    auto phi = c * (scalars->phi_bar);
    scalars->phi_bar = s * (scalars->phi_bar);
    blas::copy(num_cols, vectors->w, inc, vectors->temp, inc, queue);
    magma_queue_sync(queue);
    precond_apply(MagmaNoTrans, num_cols, precond_mtx, ld_precond, vectors->temp,
                  inc, queue);
    magma_queue_sync(queue);
    blas::axpy(num_cols, phi / rho, vectors->temp, 1, sol, 1, queue);
    magma_queue_sync(queue);

    // compute new w_vector
    blas::scale(num_cols, -(theta / rho), vectors->w, inc, queue);
    magma_queue_sync(queue);
    blas::axpy(num_cols, 1.0, vectors->v, 1, vectors->w, 1, queue);
    magma_queue_sync(queue);
}

template <typename value_type>
bool check_stopping_criteria(matrix::dense<value_type>* mtx_in,
                             matrix::dense<value_type>* rhs_in,
                             matrix::dense<value_type>* sol_in,
                             value_type* res_vector, magma_int_t max_iter,
                             double tolerance, magma_int_t* iter,
                             double* resnorm, magma_queue_t queue)
{
    auto mtx = mtx_in->get_values();
    auto rhs = rhs_in->get_values();
    auto sol = sol_in->get_values();
    auto num_rows = mtx_in->get_size()[0];
    auto num_cols = mtx_in->get_size()[1];
    *iter += 1;
    magma_int_t inc = 1;
    blas::copy(num_rows, rhs, inc, res_vector, inc, queue);
    blas::gemv(MagmaNoTrans, num_rows, num_cols, -1.0, mtx, num_rows, sol, inc,
               1.0, res_vector, inc, queue);
    auto rhsnorm = blas::norm2(num_rows, rhs, inc, queue);
    *resnorm = blas::norm2(num_rows, res_vector, inc, queue);
    *resnorm = *resnorm / rhsnorm;
    std::cout << "*resnorm: " << *resnorm << '\n';
    std::cout << "*iter: " << *iter << '\n';
    std::cout << "max_iter: " << max_iter << '\n';
    std::cout << "((*iter) >= max_iter): " << ((*iter) >= max_iter) << '\n';
    if ((*iter >= max_iter) || (*resnorm < tolerance)) {
        std::cout << "returns true\n";
        return true;
    } else {
        std::cout << "returns false\n";
        return false;
    }
}

template <typename value_type_in, typename value_type, typename index_type>
void lsqr<value_type_in, value_type, index_type>::allocate_vectors(dim2 size)
{
    memory::malloc(&vectors->u, size[0]);
    memory::malloc(&vectors->v, size[0]);
    memory::malloc(&vectors->w, size[0]);
    memory::malloc(&vectors->temp, size[0]);
    if (!std::is_same<value_type_in, value_type>::value) {
        memory::malloc(&vectors->u_in, size[0]);
        memory::malloc(&vectors->v_in, size[0]);
        memory::malloc(&vectors->temp_in, size[0]);
        memory::malloc(&vectors->mtx_in, size[0] * size[1]);
        // if (sizeof(value_type) > sizeof(value_type_in)) {
        //     cuda::demote(size[0], size[1], mtx, size[0], vectors->mtx_in,
        //     size[0]);
        // }
        // else if (sizeof(value_type) < sizeof(value_type_in)) {
        //     cuda::promote(size[0], size[1], mtx, size[0], vectors->mtx_in,
        //     size[0]);
        // }
    }
}

template <typename value_type_in, typename value_type, typename index_type>
void lsqr<value_type_in, value_type, index_type>::free_vectors()
{
    memory::free(vectors->u);
    memory::free(vectors->v);
    memory::free(vectors->w);
    memory::free(vectors->temp);
    if (!std::is_same<value_type_in, value_type>::value) {
        memory::free(vectors->u_in);
        memory::free(vectors->v_in);
        memory::free(vectors->mtx_in);
        memory::free(vectors->temp_in);
    }
}

template <typename value_type_in, typename value_type, typename index_type>
void run_lsqr(matrix::dense<value_type>* mtx, matrix::dense<value_type>* rhs,
              matrix::dense<value_type>* sol,
              preconditioner::preconditioner<value_type_in, value_type,
                                             index_type>* precond,
              temp_scalars<value_type, index_type>* scalars,
              temp_vectors<value_type_in, value_type, index_type>* vectors,
              magma_int_t max_iter, double tolerance, magma_int_t* iter,
              double* resnorm, magma_queue_t queue, double* t_solve)
{
    *t_solve = 0;
    *iter = 0;
    auto size = mtx->get_size();
    initialize(size, mtx->get_values(), rhs->get_values(), precond, scalars,
               vectors, queue, t_solve);
    double t = magma_sync_wtime(queue);
    while (1) {
        step_1(mtx, precond, scalars, vectors, queue);
        step_2(mtx, sol, precond, scalars, vectors, queue);
        if (check_stopping_criteria(mtx, rhs, sol, vectors->temp,
            max_iter, tolerance, iter, resnorm, queue)) {
            std::cout << "*iter: " << *iter << ", tolerance: " << tolerance << '\n';
            break;
        }
    }
    *t_solve += (magma_sync_wtime(queue) - t);
}

template void run_lsqr<double, double, int>(
    matrix::dense<double>* mtx, matrix::dense<double>* rhs,
    matrix::dense<double>* sol,
    preconditioner::preconditioner<double, double, int>* precond,
    temp_scalars<double, int>* scalars,
    temp_vectors<double, double, int>* vectors, magma_int_t max_iter,
    double tolerance, magma_int_t* iter, double* resnorm, magma_queue_t queue,
    double* t_solve);

template void run_lsqr<float, double, int>(
    matrix::dense<double>* mtx, matrix::dense<double>* rhs,
    matrix::dense<double>* sol,
    preconditioner::preconditioner<float, double, int>* precond,
    temp_scalars<double, int>* scalars,
    temp_vectors<float, double, int>* vectors, magma_int_t max_iter,
    double tolerance, magma_int_t* iter, double* resnorm, magma_queue_t queue,
    double* t_solve);

template void run_lsqr<__half, double, int>(
    matrix::dense<double>* mtx, matrix::dense<double>* rhs,
    matrix::dense<double>* sol,
    preconditioner::preconditioner<__half, double, int>* precond,
    temp_scalars<double, int>* scalars,
    temp_vectors<__half, double, int>* vectors, magma_int_t max_iter,
    double tolerance, magma_int_t* iter, double* resnorm, magma_queue_t queue,
    double* t_solve);

template void run_lsqr<float, float, int>(
    matrix::dense<float>* mtx, matrix::dense<float>* rhs,
    matrix::dense<float>* sol,
    preconditioner::preconditioner<float, float, int>* precond,
    temp_scalars<float, int>* scalars, temp_vectors<float, float, int>* vectors,
    magma_int_t max_iter, double tolerance, magma_int_t* iter, double* resnorm,
    magma_queue_t queue, double* t_solve);

template void run_lsqr<__half, float, int>(
    matrix::dense<float>* mtx, matrix::dense<float>* rhs,
    matrix::dense<float>* sol,
    preconditioner::preconditioner<__half, float, int>* precond,
    temp_scalars<float, int>* scalars,
    temp_vectors<__half, float, int>* vectors, magma_int_t max_iter,
    double tolerance, magma_int_t* iter, double* resnorm, magma_queue_t queue,
    double* t_solve);

}  // namespace solver
}  // namespace rls
