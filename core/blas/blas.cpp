#include <cuda_runtime.h>
#include <iostream>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../memory/magma_context.hpp"
#include "../memory/memory.hpp"


namespace rls {
namespace blas {


double norm2(magma_int_t num_rows, double* v_vector, magma_int_t inc,
             magma_queue_t queue)
{
    return magma_dnrm2(num_rows, v_vector, inc, queue);
}

float norm2(magma_int_t num_rows, float* v_vector, magma_int_t inc,
            magma_queue_t queue)
{
    return magma_snrm2(num_rows, v_vector, inc, queue);
}


void copy(magma_int_t num_rows, double* source_vector, magma_int_t inc_u,
          double* dest_vector, magma_int_t inc_v, magma_queue_t queue)
{
    magma_dcopy(num_rows, source_vector, inc_u, dest_vector, inc_v, queue);
}

void copy(magma_int_t num_rows, float* source_vector, magma_int_t inc_u,
          float* dest_vector, magma_int_t inc_v, magma_queue_t queue)
{
    magma_scopy(num_rows, source_vector, inc_u, dest_vector, inc_v, queue);
}


void scale(magma_int_t num_rows, double alpha, double* v_vector,
           magma_int_t inc, magma_queue_t queue)
{
    magma_dscal(num_rows, alpha, v_vector, inc, queue);
}

void scale(magma_int_t num_rows, float alpha, float* v_vector, magma_int_t inc,
           magma_queue_t queue)
{
    magma_sscal(num_rows, alpha, v_vector, inc, queue);
}


void axpy(magma_int_t num_rows, double alpha, double* u_vector,
          magma_int_t inc_u, double* v_vector, magma_int_t inc_v,
          magma_queue_t queue)
{
    magma_daxpy(num_rows, alpha, u_vector, inc_u, v_vector, inc_v, queue);
}

void axpy(magma_int_t num_rows, float alpha, float* u_vector, magma_int_t inc_u,
          float* v_vector, magma_int_t inc_v, magma_queue_t queue)
{
    magma_saxpy(num_rows, alpha, u_vector, inc_u, v_vector, inc_v, queue);
}


void gemv(magma_trans_t trans, magma_int_t num_rows, magma_int_t num_cols,
          double alpha, double* mtx, magma_int_t ld, double* u_vector,
          magma_int_t inc_u, double beta, double* v_vector, magma_int_t inc_v,
          magma_queue_t queue)
{
    magma_dgemv(trans, num_rows, num_cols, alpha, mtx, num_rows, u_vector,
                inc_u, beta, v_vector, inc_v, queue);
}

void gemv(magma_trans_t trans, magma_int_t num_rows, magma_int_t num_cols,
          float alpha, float* mtx, magma_int_t ld, float* u_vector,
          magma_int_t inc_u, float beta, float* v_vector, magma_int_t inc_v,
          magma_queue_t queue)
{
    magmablas_sgemv(trans, num_rows, num_cols, alpha, mtx, ld, u_vector, inc_u,
                    beta, v_vector, inc_v, queue);
}

void gemv(magma_trans_t trans, magma_int_t num_rows, magma_int_t num_cols,
          magmaHalf alpha, magmaHalf_ptr mtx, magma_int_t ld,
          magmaHalf_ptr u_vector, magma_int_t inc_u, magmaHalf beta,
          magmaHalf_ptr v_vector, magma_int_t inc_v, magma_queue_t queue)
{
    if (trans == MagmaNoTrans) {
        magma_hgemm(MagmaNoTrans, MagmaNoTrans, num_rows, 1, num_cols, alpha,
                    mtx, ld, u_vector, ld, beta, v_vector, ld, queue);
    } else if (trans == MagmaTrans) {
        magma_hgemm(MagmaTrans, MagmaNoTrans, num_cols, 1, num_rows, alpha, mtx,
                    ld, u_vector, ld, beta, v_vector, ld, queue);
    }
}


void trsv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaDouble_const_ptr dA, magma_int_t ldda,
          magmaDouble_ptr dx, magma_int_t incx, magma_queue_t queue)
{
    magma_dtrsv(uplo, trans, diag, n, dA, ldda, dx, incx, queue);
}

void trsv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaFloat_const_ptr dA, magma_int_t ldda,
          magmaFloat_ptr dx, magma_int_t incx, magma_queue_t queue)
{
    magma_strsv(uplo, trans, diag, n, dA, ldda, dx, incx, queue);
}

void trsm(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
          magma_diag_t diag, magma_int_t m, magma_int_t n,
          double alpha, magmaDouble_const_ptr dA,
          magma_int_t ldda, magmaDouble_ptr dB,
          magma_int_t lddb, magma_queue_t queue)	
{
    magma_dtrsm(side, uplo, trans, diag, m, n, alpha, dA,
                ldda, dB, lddb, queue)	;
}

void trsm(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
          magma_diag_t diag, magma_int_t m, magma_int_t n,
          float alpha, magmaFloat_const_ptr dA,
          magma_int_t ldda, magmaFloat_ptr dB,
          magma_int_t lddb, magma_queue_t queue)	
{
    magma_strsm(side, uplo, trans, diag, m, n, alpha, dA,
                ldda, dB, lddb, queue)	;
}



void trmv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaDouble_const_ptr dA, magma_int_t ldda,
          magmaDouble_ptr dx, magma_int_t incx, magma_queue_t queue)
{
    magma_dtrmv(uplo, trans, diag, n, dA, ldda, dx, incx, queue);
}

void trmv(magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaFloat_const_ptr dA, magma_int_t ldda,
          magmaFloat_ptr dx, magma_int_t incx, magma_queue_t queue)
{
    magma_strmv(uplo, trans, diag, n, dA, ldda, dx, incx, queue);
}

magma_int_t trtri(magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
                  magmaDouble_ptr dA, magma_int_t ldda, magma_int_t* info) {
    return magma_dtrtri_gpu(uplo, diag, n, dA, ldda, info);
}

magma_int_t trtri(magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
                  magmaFloat_ptr dA, magma_int_t ldda, magma_int_t* info) {
    return magma_strtri_gpu(uplo, diag, n, dA, ldda, info);
}


void trmm(magma_side_t side, magma_uplo_t uplo,
         magma_trans_t trans, magma_diag_t diag, magma_int_t m,
         magma_int_t n, double alpha, magmaDouble_const_ptr dA,
         magma_int_t ldda, magmaDouble_ptr dB, magma_int_t lddb,
         magma_queue_t queue) {
    magma_dtrmm(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
}


void trmm(magma_side_t side, magma_uplo_t uplo,
         magma_trans_t trans, magma_diag_t diag, magma_int_t m,
         magma_int_t n, float alpha, magmaFloat_const_ptr dA,
         magma_int_t ldda, magmaFloat_ptr dB, magma_int_t lddb,
         magma_queue_t queue) {
    magma_strmm(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
}

// template <ContextType device_type = CUDA>
// void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
        //   magma_int_t n, magma_int_t k, double alpha, magmaDouble_const_ptr dA,
        //   magma_int_t ldda, magmaDouble_const_ptr dB, magma_int_t lddb,
        //   double beta, magmaDouble_ptr dC, magma_int_t lddc,
        //   Context<device_type>* context)
// {
    // magma_dgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                // lddc, context->get_queue());
// }
// 
// template <ContextType device_type = CUDA>
// void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
        //   magma_int_t n, magma_int_t k, float alpha, magmaFloat_const_ptr dA,
        //   magma_int_t ldda, magmaFloat_const_ptr dB, magma_int_t lddb,
        //   float beta, magmaFloat_ptr dC, magma_int_t lddc,
        //   Context<device_type>* context)
// {
    // magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                // lddc, context->get_queue());
// }
// 
// template <ContextType device_type = CUDA>
// void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
        //   magma_int_t n, magma_int_t k, magmaHalf alpha, magmaHalf_const_ptr dA,
        //   magma_int_t ldda, magmaHalf_const_ptr dB, magma_int_t lddb,
        //   magmaHalf beta, magmaHalf_ptr dC, magma_int_t lddc,
        //   Context<device_type>* context)
// {
    // magma_hgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                // lddc, context->get_queue());
// }

void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, double alpha, magmaDouble_const_ptr dA,
          magma_int_t ldda, magmaDouble_const_ptr dB, magma_int_t lddb,
          double beta, magmaDouble_ptr dC, magma_int_t lddc,
          Context<CUDA>* context)
{
    magma_dgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                lddc, context->get_queue());
}

void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, float alpha, magmaFloat_const_ptr dA,
          magma_int_t ldda, magmaFloat_const_ptr dB, magma_int_t lddb,
          float beta, magmaFloat_ptr dC, magma_int_t lddc,
          Context<CUDA>* context)
{
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                lddc, context->get_queue());
}

void gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, magmaHalf alpha, magmaHalf_const_ptr dA,
          magma_int_t ldda, magmaHalf_const_ptr dB, magma_int_t lddb,
          magmaHalf beta, magmaHalf_ptr dC, magma_int_t lddc,
          Context<CUDA>* context)
{
    magma_hgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                lddc, context->get_queue());
}

magma_int_t geqrf2_gpu(magma_int_t m, magma_int_t n, magmaDouble_ptr dA,
                       magma_int_t ldda, double* tau, magma_int_t* info)
{
    return magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
}

magma_int_t geqrf2_gpu(magma_int_t m, magma_int_t n, magmaFloat_ptr dA,
                       magma_int_t ldda, float* tau, magma_int_t* info)
{
    return magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
}


double dot(magma_int_t n, magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy, magma_queue_t queue)	
{
    return magma_ddot(n, dx, incx, dy, incy, queue);
}

float dot(magma_int_t n, magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy, magma_queue_t queue)	
{
    return magma_sdot(n, dx, incx, dy, incy, queue);
}

float dot(magma_int_t n, const __half* dx, magma_int_t incx,
    const __half* dy, magma_int_t incy, magma_queue_t queue)	
{
    // __half* dc;
    // memory::malloc(&dc, 1);
    // magma_hgemm(MagmaNoTrans, MagmaNoTrans, 1, 1, n, 1.0, dx, n, dy, n, 1.0, dC,
                // 1, queue);
    // 
    // memory::free(dc);
    // return magma_sdot(n, dx, incx, dy, incy, queue);
}




}  // namespace blas
}  // namespace rls
