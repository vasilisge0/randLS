#include <cuda_runtime.h>
#include <iostream>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../blas/blas.hpp"
#include "../memory/memory.hpp"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "base_types.hpp"
#include "lsqr.hpp"
#include <cmath>
#include "../../utils/io.hpp"


namespace rls {
namespace solver {
namespace lsqr {
namespace {

void getmatrix(magma_int_t num_rows, magma_int_t num_cols, double* source_vector, magma_int_t ld_source, double* dest_vector, magma_int_t ld_dest, magma_queue_t queue)
{
    magma_dgetmatrix(num_rows, num_cols, source_vector, ld_source,
                     dest_vector, ld_source, queue);
}

void getmatrix(magma_int_t num_rows, magma_int_t num_cols, float* source_vector, magma_int_t ld_source, float* dest_vector, magma_int_t ld_dest, magma_queue_t queue)
{
    magma_sgetmatrix(num_rows, num_cols, source_vector, ld_source,
                     dest_vector, ld_source, queue);
}

void precond_apply(magma_trans_t trans, magma_int_t num_rows,
                   double* precond_mtx, magma_int_t ld, double* u_vector,
                   magma_int_t inc_u, magma_queue_t queue)
{
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
    std::cout << "*beta (norm): " << *beta << '\n';
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
void initialize(index_type num_rows, index_type num_cols, value_type* mtx, value_type* rhs, index_type* iter,
                temp_scalars<value_type, index_type>& scalars,
                temp_vectors<value_type_in, value_type, index_type>& vectors,
                magma_queue_t queue)
{
    vectors.inc = 1;
    memory::malloc(&vectors.u, num_rows);
    memory::malloc(&vectors.v, num_rows);
    memory::malloc(&vectors.w, num_rows);
    memory::malloc(&vectors.temp, num_rows);
    if (!std::is_same<value_type_in, value_type>::value) {
        memory::malloc(&vectors.u_in, num_rows);
        memory::malloc(&vectors.v_in, num_rows);
        memory::malloc(&vectors.temp_in, num_rows);
        memory::malloc(&vectors.mtx_in, num_rows * num_cols);

        if (sizeof(value_type) > sizeof(value_type_in)) {
            cuda::demote(num_rows, num_cols, mtx, num_rows, vectors.mtx_in, num_rows);
        }
        else if (sizeof(value_type) < sizeof(value_type_in)) {
            cuda::promote(num_rows, num_cols, mtx, num_rows, vectors.mtx_in, num_rows);
        }
    }

    *iter = 0;
    scalars.beta = blas::norm2(num_rows, rhs, vectors.inc, queue);
    blas::copy(num_rows, rhs, vectors.inc, vectors.u, vectors.inc, queue);
    blas::scale(num_rows, 1 / scalars.beta, vectors.u, vectors.inc, queue);
    blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, vectors.u,
               vectors.inc, 1.0, vectors.v, vectors.inc, queue);

    scalars.alpha = blas::norm2(num_cols, vectors.v, vectors.inc, queue);
    blas::scale(num_cols, 1 / scalars.alpha, vectors.v, vectors.inc, queue);
    blas::copy(num_cols, vectors.v, vectors.inc, vectors.w, vectors.inc, queue);
    scalars.rho_bar = scalars.alpha;
    scalars.phi_bar = scalars.beta;
}


template <typename value_type_in, typename value_type, typename index_type>
void initialize(index_type num_rows, index_type num_cols, value_type* mtx, value_type* rhs,
                value_type* precond_mtx, index_type ld_precond,
                index_type* iter, 
                temp_scalars<value_type, index_type>& scalars,
                temp_vectors<value_type_in, value_type, index_type>& vectors,
                magma_queue_t queue)
{
    vectors.inc = 1;
    memory::malloc(&vectors.u, num_rows);
    memory::malloc(&vectors.v, num_rows);
    memory::malloc(&vectors.w, num_rows);
    memory::malloc(&vectors.temp, num_rows);
    if (!std::is_same<value_type_in, value_type>::value) {
        memory::malloc(&vectors.u_in, num_rows);
        memory::malloc(&vectors.v_in, num_rows);
        memory::malloc(&vectors.temp_in, num_rows);
        memory::malloc(&vectors.mtx_in, num_rows * num_cols);
        std::cout << "testing...\n";

        // if (sizeof(value_type) > sizeof(value_type_in)) {
            std::cout << "IN INITIALIZE\n" << '\n';
            cuda::demote(num_rows, num_cols, mtx, num_rows, vectors.mtx_in, num_rows);
        // }
        // else if (sizeof(value_type) < sizeof(value_type_in)) {
        //     cuda::promote(num_rows, num_cols, mtx, num_rows, vectors.mtx_in, num_rows);
        // }
    }

    *iter = 0;
    scalars.beta = blas::norm2(num_rows, rhs, vectors.inc, queue);
    std::cout <<" here\n";
    // io::print_mtx_gpu(10, 1, rhs, num_rows, queue);
    
    blas::copy(num_rows, rhs, vectors.inc, vectors.u, vectors.inc, queue);
    blas::scale(num_rows, 1 / scalars.beta, vectors.u, vectors.inc, queue);
    std::cout << ">>> beta: " << scalars.beta << '\n';

    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors.u, num_rows,
                     vectors.u_in, num_rows);
        cuda::demote(num_rows, 1, vectors.v, num_rows, vectors.v_in,
                     num_rows);
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors.mtx_in, num_rows, vectors.u_in,
               vectors.inc, 0.0, vectors.v_in, vectors.inc, queue);
        cuda::promote(num_rows, 1, vectors.v_in,
            num_rows, vectors.v, num_rows);
    }
    else {
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, vectors.u,
               vectors.inc, 0.0, vectors.v, vectors.inc, queue);
    }

    precond_apply(MagmaTrans, num_cols, precond_mtx, ld_precond, vectors.v,
                  vectors.inc, queue);
    scalars.alpha = blas::norm2(num_cols, vectors.v, vectors.inc, queue);
    std::cout << ">>> alpha: " << scalars.alpha << '\n';

    blas::scale(num_cols, 1 / scalars.alpha, vectors.v, vectors.inc, queue);
    blas::copy(num_cols, vectors.v, vectors.inc, vectors.w, vectors.inc, queue);
    scalars.phi_bar = scalars.beta;
    scalars.rho_bar = scalars.alpha;
}

template <typename value_type_in, typename value_type, typename index_type>
void finalize(temp_vectors<value_type_in, value_type, index_type>& vectors) {
    memory::free(vectors.u);
    memory::free(vectors.v);
    memory::free(vectors.w);
    memory::free(vectors.temp);
    if (!std::is_same<value_type_in, value_type>::value) {
        memory::free(vectors.u_in);
        memory::free(vectors.v_in);
        memory::free(vectors.mtx_in);
        memory::free(vectors.temp_in);
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
            temp_scalars<value_type, index_type>& scalars,
            temp_vectors<value_type_in, value_type, index_type>& vectors,
            magma_queue_t queue)
{
    index_type inc = 1;
    // compute new u_vector
    blas::scale(num_rows, scalars.alpha, vectors.u, inc, queue);
    blas::copy(num_cols, vectors.v, inc, vectors.temp, inc, queue);
    precond_apply(MagmaNoTrans, num_cols, precond_mtx, ld_precond, vectors.temp,
                  inc, queue);
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors.temp, num_rows, vectors.temp_in, num_rows);
        cuda::demote(num_rows, 1, vectors.u, num_rows, vectors.u_in, num_rows);
        cudaDeviceSynchronize();
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, vectors.mtx_in, num_rows,
                   vectors.temp_in, inc, -1.0, vectors.u_in, inc, queue);
        cuda::promote(num_rows, 1, vectors.u_in, num_rows, vectors.u, num_rows);
        cudaDeviceSynchronize();
    }
    else {
        blas::gemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows, vectors.temp,
               inc, -1.0, vectors.u, inc, queue);
    }
    scalars.beta = blas::norm2(num_rows, vectors.u, inc, queue);
    blas::scale(num_rows, 1 / scalars.beta, vectors.u, inc, queue);

    // compute new v_vector
    if (!std::is_same<value_type_in, value_type>::value) {
        cuda::demote(num_rows, 1, vectors.u, num_rows, vectors.u_in,
                     num_rows);
        cuda::demote(num_rows, 1, vectors.temp, num_rows,
                     vectors.temp_in, num_rows);
        cudaDeviceSynchronize();
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, vectors.mtx_in, num_rows, vectors.u_in,
               inc, 0.0, vectors.temp_in, inc, queue);
        cuda::promote(num_rows, 1, vectors.temp_in,
            num_rows, vectors.temp, num_rows);   
        cudaDeviceSynchronize();
    }
    else {
        blas::gemv(MagmaTrans, num_rows, num_cols, 1.0, mtx, num_rows, vectors.u,
               inc, 0.0, vectors.temp, inc, queue);
    }
    
    precond_apply(MagmaTrans, num_cols, precond_mtx, ld_precond, vectors.temp,
                  inc, queue);
    blas::axpy(num_cols, -(scalars.beta), vectors.v, 1, vectors.temp, 1, queue);
    scalars.alpha = blas::norm2(num_cols, vectors.temp, inc, queue);

    blas::scale(num_cols, 1 / scalars.alpha, vectors.temp, inc, queue);
    blas::copy(num_cols, vectors.temp, inc, vectors.v, inc, queue);
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
            value_type* rhs, value_type* sol,
            value_type* precond_mtx, index_type ld_precond,
            temp_scalars<value_type, index_type>& scalars,
            temp_vectors<value_type_in, value_type, index_type>& vectors,
            magma_queue_t queue)
{
    index_type inc = 1;
    auto rho = std::sqrt(((scalars.rho_bar) * (scalars.rho_bar) + scalars.beta * scalars.beta));
    auto c = (scalars.rho_bar) / rho;
    auto s = scalars.beta / rho;
    auto theta = s * scalars.alpha;
    scalars.rho_bar = -c * scalars.alpha;
    auto phi = c * (scalars.phi_bar);
    scalars.phi_bar = s * (scalars.phi_bar);
    blas::copy(num_cols, vectors.w, inc, vectors.temp, inc, queue);
    precond_apply(MagmaNoTrans, num_cols, precond_mtx, ld_precond, vectors.temp,
                  inc, queue);
    blas::axpy(num_cols, phi / rho, vectors.temp, 1, sol, 1, queue);
    // compute new w_vector
    blas::scale(num_cols, -(theta / rho), vectors.w, inc, queue);
    blas::axpy(num_cols, 1.0, vectors.v, 1, vectors.w, 1, queue);
}

template <typename value_type, typename index_type>
bool check_stopping_criteria(index_type num_rows, index_type num_cols,
                             value_type* mtx, value_type* rhs, value_type* sol,
                             value_type* res_vector, index_type* iter,
                             index_type max_iter, value_type max_true_relres,
                             double* resnorm, magma_queue_t queue)
{
    *iter += 1;
    index_type inc = 1;
    blas::copy(num_rows, rhs, inc, res_vector, inc, queue);
    blas::gemv(MagmaNoTrans, num_rows, num_cols, -1.0, mtx, num_rows, sol, inc,
               1.0, res_vector, inc, queue);
    auto rhsnorm = blas::norm2(num_rows, rhs, inc, queue);
    *resnorm = blas::norm2(num_rows, res_vector, inc, queue);
    *resnorm = *resnorm / rhsnorm;
    if ((*iter >= max_iter) || (*resnorm < max_true_relres)) {
        return true;
    } else {
        return false;
    }
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

template <typename value_type, typename index_type>
void run(index_type num_rows, index_type num_cols, value_type* mtx,
          value_type* rhs, value_type* init_sol, value_type* sol,
          index_type max_iter, index_type* iter, value_type tol,
          double* resnorm, magma_queue_t queue)
{
    // temp_scalars<value_type, index_type> scalars;
    // temp_vectors<value_type_in, value_type, index_type> vectors;

    value_type* u_vector = nullptr;
    value_type* v_vector = nullptr;
    value_type* w_vector = nullptr;
    value_type* tmp_vector = nullptr;
    value_type alpha = 0.0;
    value_type beta = 0.0;
    value_type rho_bar = 0.0;
    value_type phi_bar = 0.0;
    allocate_memory(num_rows, num_cols, &u_vector, &v_vector, &w_vector,
                    &tmp_vector);
    initialize(num_rows, num_cols, iter, &alpha, &beta, &rho_bar,
                       &phi_bar, mtx, u_vector, v_vector, w_vector, rhs, queue);
    std::cout << "alpha: " << alpha << ", beta: " << beta << '\n';                   
    while (1) {
        step_1(num_rows, num_cols, &alpha, &beta, mtx, u_vector, v_vector,
               queue);
        step_2(num_rows, num_cols, alpha, beta, mtx, &rho_bar, &phi_bar,
               u_vector, v_vector, w_vector, rhs, sol, queue);
        if (check_stopping_criteria(num_rows, num_cols, mtx, rhs, sol,
                                    tmp_vector, iter, max_iter, tol, resnorm,
                                    queue)) {
            break;
        }
    }
    free_memory(u_vector, v_vector, w_vector, tmp_vector);
}

template void run<double, magma_int_t>(magma_int_t num_rows,
                                        magma_int_t num_cols, double* mtx,
                                        double* rhs, double* init_sol,
                                        double* sol, magma_int_t max_iter,
                                        magma_int_t* iter, double tol,
                                        double* resnorm, magma_queue_t queue);

template void run<float, magma_int_t>(magma_int_t num_rows,
                                       magma_int_t num_cols, float* mtx,
                                       float* rhs, float* init_sol, float* sol,
                                       magma_int_t max_iter, magma_int_t* iter,
                                       float tol, double* resnorm,
                                       magma_queue_t queue);

template <typename value_type_in, typename value_type, typename index_type>
void run(index_type num_rows, index_type num_cols, value_type* mtx,
          value_type* rhs, value_type* init_sol, value_type* sol,
          index_type max_iter, index_type* iter, value_type tol,
          double* resnorm, value_type* precond_mtx, index_type ld_precond,
          magma_queue_t queue, double* t_solve)
{
    temp_scalars<value_type, index_type> scalars;
    temp_vectors<value_type_in, value_type, index_type> vectors;
    value_type* u_vector = nullptr;
    value_type* v_vector = nullptr;
    value_type* w_vector = nullptr;
    value_type* tmp_vector = nullptr;
    value_type alpha = 0.0;
    value_type beta = 0.0;
    value_type rho_bar = 0.0;
    value_type phi_bar = 0.0;
    initialize(num_rows, num_cols, mtx, rhs,
               precond_mtx, ld_precond, iter, scalars, vectors, queue);
    *t_solve = 0;
    #if PRINT_RELRES
        std::cout << "iter: " << *iter << ", resnorm: " << *resnorm
                  << ", alpha: " << scalars.alpha << ", beta: " << scalars.beta << '\n';
    #endif
    double t = magma_sync_wtime(queue);
    while (1) {
        step_1(num_rows, num_cols, mtx, precond_mtx, ld_precond, scalars, vectors, queue);
        step_2(num_rows, num_cols, mtx, rhs, sol, precond_mtx, ld_precond,
               scalars, vectors, queue);
        if (check_stopping_criteria(num_rows, num_cols, mtx, rhs, sol,
                                    vectors.temp, iter, max_iter, tol, resnorm,
                                    queue)) {
            break;
        }
        #if PRINT_RELRES
            std::cout << "iter: " << *iter << ", resnorm: " << *resnorm
                      << ", alpha: " << scalars.alpha << ", beta: " << scalars.beta << '\n';
        #endif
    }
    *t_solve += (magma_sync_wtime(queue) - t);
    finalize(vectors);
}

template void run<double, double, magma_int_t>(
    magma_int_t num_rows, magma_int_t num_cols, double* mtx, double* rhs,
    double* init_sol, double* sol, magma_int_t max_iter, magma_int_t* iter,
    double tol, double* resnorm, double* precond_mtx, magma_int_t ld_precond,
    magma_queue_t queue, double* t_solve);
    
template void run<float, double, magma_int_t>(
    magma_int_t num_rows, magma_int_t num_cols, double* mtx, double* rhs,
    double* init_sol, double* sol, magma_int_t max_iter, magma_int_t* iter,
    double tol, double* resnorm, double* precond_mtx, magma_int_t ld_precond,
    magma_queue_t queue, double* t_solve);

template void run<__half, double, magma_int_t>(
    magma_int_t num_rows, magma_int_t num_cols, double* mtx, double* rhs,
    double* init_sol, double* sol, magma_int_t max_iter, magma_int_t* iter,
    double tol, double* resnorm, double* precond_mtx, magma_int_t ld_precond,
    magma_queue_t queue, double* t_solve);    

template void run<float, float, magma_int_t>(
    magma_int_t num_rows, magma_int_t num_cols, float* mtx, float* rhs,
    float* init_sol, float* sol, magma_int_t max_iter, magma_int_t* iter,
    float tol, double* resnorm, float* precond_mtx, magma_int_t ld_precond,
    magma_queue_t queue, double* t_solve);

template void run<__half, float, magma_int_t>(
    magma_int_t num_rows, magma_int_t num_cols, float* mtx, float* rhs,
    float* init_sol, float* sol, magma_int_t max_iter, magma_int_t* iter,
    float tol, double* resnorm, float* precond_mtx, magma_int_t ld_precond,
    magma_queue_t queue, double* t_solve);


}  // namespace lsqr
}  // namespace solver
}  // namespace rls
