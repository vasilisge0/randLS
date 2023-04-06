#ifndef RANDLS_BLAS_HPP
#define RANDLS_BLAS_HPP


#include "magma_v2.h"
#include "../memory/magma_context.hpp"


namespace rls {
namespace blas {


double norm2(magma_int_t num_rows, double* v_vector, magma_int_t inc,
             magma_queue_t queue);

float norm2(magma_int_t num_rows, float* v_vector, magma_int_t inc,
            magma_queue_t queue);

void copy(magma_int_t num_rows, double* source_vector, magma_int_t inc,
          double* dest_vector, magma_int_t inc_v, magma_queue_t queue);

void copy(magma_int_t num_rows, float* source_vector, magma_int_t inc_u,
          float* dest_vector, magma_int_t inc_v, magma_queue_t queue);

void scale(magma_int_t num_rows, double alpha, double* v_vector,
           magma_int_t inc, magma_queue_t queue);

void scale(magma_int_t num_rows, float alpha, float* v_vector, magma_int_t inc,
           magma_queue_t queue);

void axpy(magma_int_t num_rows, double alpha, double* u_vector,
          magma_int_t inc_u, double* v_vector, magma_int_t inc_v,
          magma_queue_t queue);

void axpy(magma_int_t num_rows, float alpha, float* u_vector, magma_int_t inc_u,
          float* v_vector, magma_int_t inc_v, magma_queue_t queue);

void gemv(magma_trans_t trans, magma_int_t num_rows, magma_int_t num_cols,
          double alpha, double* mtx, magma_int_t ld, double* u_vector,
          magma_int_t inc_u, double beta, double* v_vector, magma_int_t inc_v,
          magma_queue_t queue);

void gemv(magma_trans_t trans, magma_int_t num_rows, magma_int_t num_cols,
          float alpha, float* mtx, magma_int_t ld, float* u_vector,
          magma_int_t inc_u, float beta, float* v_vector, magma_int_t inc_v,
          magma_queue_t queue);

void gemv(magma_trans_t trans, magma_int_t num_rows, magma_int_t num_cols,
          magmaHalf alpha, magmaHalf_ptr mtx, magma_int_t ld,
          magmaHalf_ptr u_vector, magma_int_t inc_u, magmaHalf beta,
          magmaHalf_ptr v_vector, magma_int_t inc_v, magma_queue_t queue);

void trsv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaDouble_const_ptr dA, magma_int_t ldda,
          magmaDouble_ptr dx, magma_int_t incx, magma_queue_t queue);

void trsv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaFloat_const_ptr dA, magma_int_t ldda,
          magmaFloat_ptr dx, magma_int_t incx, magma_queue_t queue);

void trmv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaDouble_const_ptr dA, magma_int_t ldda,
          magmaDouble_ptr dx, magma_int_t incx, magma_queue_t queue);

void trmv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaFloat_const_ptr dA, magma_int_t ldda,
          magmaFloat_ptr dx, magma_int_t incx, magma_queue_t queue);

void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, double alpha, magmaDouble_const_ptr dA,
          magma_int_t ldda, magmaDouble_const_ptr dB, magma_int_t lddb,
          double beta, magmaDouble_ptr dC, magma_int_t lddc,
          Context* context);

void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, float alpha, magmaFloat_const_ptr dA,
          magma_int_t ldda, magmaFloat_const_ptr dB, magma_int_t lddb,
          float beta, magmaFloat_ptr dC, magma_int_t lddc,
          Context* context);

void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, magmaHalf alpha, magmaHalf_const_ptr dA,
          magma_int_t ldda, magmaHalf_const_ptr dB, magma_int_t lddb,
          magmaHalf beta, magmaHalf_ptr dC, magma_int_t lddc,
          Context* context);

magma_int_t geqrf2_gpu(magma_int_t m, magma_int_t n, magmaDouble_ptr dA,
                       magma_int_t ldda, double* tau, magma_int_t* info);

magma_int_t geqrf2_gpu(magma_int_t m, magma_int_t n, magmaFloat_ptr dA,
                       magma_int_t ldda, float* tau, magma_int_t* info);

double dot(magma_int_t n, magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy, magma_queue_t queue);

float dot(magma_int_t n, magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy, magma_queue_t queue);

float dot(magma_int_t n, const __half* dx, magma_int_t incx,
    const __half* dy, magma_int_t incy, magma_queue_t queue);

}  // namespace blas
}  // namespace rls

#endif
