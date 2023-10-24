#ifndef RANDLS_BLAS_HPP
#define RANDLS_BLAS_HPP


#include <iostream>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../memory/magma_context.hpp"


namespace rls {
namespace blas {


template<ContextType device_type, typename value_type, typename index_type>
value_type norm2(std::shared_ptr<Context<device_type>> context, index_type num_rows,
                 value_type* v_vector, index_type inc);

template<>
inline double norm2(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
             double* v_vector, magma_int_t inc)
{
    return magma_dnrm2(num_rows, v_vector, inc, context->get_queue());
}

template<>
inline float norm2(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
            float* v_vector, magma_int_t inc)
{
    return magma_snrm2(num_rows, v_vector, inc, context->get_queue());
}

template <ContextType device_type, typename value_type, typename index_type>
void copy(std::shared_ptr<Context<device_type>> context, index_type num_rows,
          value_type* source_vector, index_type inc_u,
          value_type* dest_vector, index_type inc_v);

template<>
inline void copy(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
          double* source_vector, magma_int_t inc_u,
          double* dest_vector, magma_int_t inc_v)
{
    magma_dcopy(num_rows, source_vector, inc_u, dest_vector, inc_v,
        context->get_queue());
}

template<>
inline void copy(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
          float* source_vector, magma_int_t inc_u, float* dest_vector,
          magma_int_t inc_v)
{
    magma_scopy(num_rows, source_vector, inc_u, dest_vector, inc_v,
        context->get_queue());
}

template<>
inline void copy(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
          __half* source_vector, magma_int_t inc_u, __half* dest_vector,
          magma_int_t inc_v)
{
    //magma_scopy(num_rows, source_vector, inc_u, dest_vector, inc_v,
    //    context->get_queue());
}

template<>
inline void copy(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows,
          double* source_vector, magma_int_t inc_u,
          double* dest_vector, magma_int_t inc_v)
{
    std::memcpy(dest_vector, source_vector, sizeof(*source_vector)*num_rows);
}

template<>
inline void copy(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows,
          float* source_vector, magma_int_t inc_u, float* dest_vector,
          magma_int_t inc_v)
{
    std::memcpy(dest_vector, source_vector, sizeof(*source_vector)*num_rows);
}

template<>
inline void copy(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows,
          __half* source_vector, magma_int_t inc_u, __half* dest_vector,
          magma_int_t inc_v)
{
    //magma_scopy(num_rows, source_vector, inc_u, dest_vector, inc_v,
    //    context->get_queue());
}

template<ContextType device_type, typename value_type, typename index_type>
inline void scale(std::shared_ptr<Context<device_type>> context, index_type num_rows,
           value_type alpha, value_type* v_vector, index_type inc);

template<>
inline void scale(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
           double alpha, double* v_vector, magma_int_t inc)
{
    magma_dscal(num_rows, alpha, v_vector, inc, context->get_queue());
}

template<>
inline void scale(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
           float alpha, float* v_vector, magma_int_t inc)
{
    magma_sscal(num_rows, alpha, v_vector, inc, context->get_queue());
}

template<ContextType device_type, typename value_type, typename index_type>
inline void axpy(std::shared_ptr<Context<device_type>> context, index_type num_rows,
          value_type alpha, value_type* u_vector, index_type inc_u,
          value_type* v_vector, index_type inc_v);

template<>
inline void axpy(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
          double alpha, double* u_vector, magma_int_t inc_u, double* v_vector,
          magma_int_t inc_v)
{
    magma_daxpy(num_rows, alpha, u_vector, inc_u, v_vector, inc_v,
        context->get_queue());
}

template<>
inline void axpy(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
          float alpha, float* u_vector, magma_int_t inc_u,
          float* v_vector, magma_int_t inc_v)
{
    magma_saxpy(num_rows, alpha, u_vector, inc_u, v_vector, inc_v,
        context->get_queue());
}

template<>
inline void axpy(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows,
          __half alpha, __half* u_vector, magma_int_t inc_u,
          __half* v_vector, magma_int_t inc_v)
{
    //magma_saxpy(num_rows, alpha, u_vector, inc_u, v_vector, inc_v,
    //    context->get_queue());
}

template<ContextType device_type, typename value_type, typename index_type>
void gemv(std::shared_ptr<Context<device_type>> context, magma_trans_t trans,
          index_type num_rows, index_type num_cols,
          value_type alpha, value_type* mtx, index_type ld, value_type* u_vector,
          index_type inc_u, value_type beta, value_type* v_vector, index_type inc_v);

template<>
inline void gemv(std::shared_ptr<Context<CUDA>> context, magma_trans_t trans,
          magma_int_t num_rows, magma_int_t num_cols,
          double alpha, double* mtx, magma_int_t ld, double* u_vector,
          magma_int_t inc_u, double beta, double* v_vector,
          magma_int_t inc_v)
{
    cudaDeviceSynchronize();
    magma_dgemv(trans, num_rows, num_cols, alpha, mtx, ld, u_vector,
                inc_u, beta, v_vector, inc_v, context->get_queue());
    //magma_dgemv(trans, num_rows, num_cols, alpha, mtx, ld, u_vector,
    //            inc_u, beta, v_vector, inc_v, context->get_queue());
    cudaDeviceSynchronize();
}

template<>
inline void gemv(std::shared_ptr<Context<CUDA>> context, magma_trans_t trans,
          magma_int_t num_rows, magma_int_t num_cols,
          float alpha, float* mtx, magma_int_t ld, float* u_vector,
          magma_int_t inc_u, float beta, float* v_vector, magma_int_t inc_v)
{
    magmablas_sgemv(trans, num_rows, num_cols, alpha, mtx, ld, u_vector,
                    inc_u, beta, v_vector, inc_v, context->get_queue());
}

template<>
inline void gemv(std::shared_ptr<Context<CUDA>> context, magma_trans_t trans,
          magma_int_t num_rows, magma_int_t num_cols,
          magmaHalf alpha, magmaHalf_ptr mtx, magma_int_t ld,
          magmaHalf_ptr u_vector, magma_int_t inc_u, magmaHalf beta,
          magmaHalf_ptr v_vector, magma_int_t inc_v)
{
    if (trans == MagmaNoTrans) {
        magma_hgemm(MagmaNoTrans, MagmaNoTrans, num_rows, 1, num_cols, alpha,
                    mtx, ld, u_vector, ld, beta, v_vector, ld,
                    context->get_queue());
    } else if (trans == MagmaTrans) {
        magma_hgemm(MagmaTrans, MagmaNoTrans, num_cols, 1, num_rows, alpha, mtx,
                    ld, u_vector, ld, beta, v_vector, ld, context->get_queue());
    }
}

template <ContextType device_type, typename value_type, typename index_type>
void trsv(std::shared_ptr<Context<device_type>> context, magma_uplo_t uplo,
          magma_trans_t trans, magma_diag_t diag,
          index_type n, const value_type* dA, index_type ldda,
          value_type* dx, index_type incx);

template<>
inline void trsv(std::shared_ptr<Context<CUDA>> context, magma_uplo_t uplo,
          magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaDouble_const_ptr dA, magma_int_t ldda,
          magmaDouble_ptr dx, magma_int_t incx)
{
    magma_dtrsv(uplo, trans, diag, n, dA, ldda, dx, incx,
        context->get_queue());
}

template<>
inline void trsv(std::shared_ptr<Context<CUDA>> context,
          magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaFloat_const_ptr dA, magma_int_t ldda,
          magmaFloat_ptr dx, magma_int_t incx)
{
    magma_strsv(uplo, trans, diag, n, dA, ldda, dx, incx,
        context->get_queue());
}

template <ContextType device_type, typename value_type, typename index_type>
void trsm(std::shared_ptr<Context<device_type>> context,
          magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
          magma_diag_t diag, index_type m, index_type n,
          value_type alpha, const value_type* dA,
          index_type ldda, value_type* dB,
          index_type lddb);

template<>
inline void trsm(std::shared_ptr<Context<CUDA>> context,
          magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
          magma_diag_t diag, magma_int_t m, magma_int_t n,
          double alpha, magmaDouble_const_ptr dA,
          magma_int_t ldda, magmaDouble_ptr dB,
          magma_int_t lddb)	
{
    magma_dtrsm(side, uplo, trans, diag, m, n, alpha, dA,
                ldda, dB, lddb, context->get_queue());
}

template<>
inline void trsm(std::shared_ptr<Context<CUDA>> context,
          magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
          magma_diag_t diag, magma_int_t m, magma_int_t n,
          float alpha, magmaFloat_const_ptr dA,
          magma_int_t ldda, magmaFloat_ptr dB,
          magma_int_t lddb)	
{
    magma_strsm(side, uplo, trans, diag, m, n, alpha, dA,
                ldda, dB, lddb, context->get_queue());
}


template <ContextType device_type, typename value_type, typename index_type>
void trmv(std::shared_ptr<Context<device_type>> context, magma_uplo_t uplo,
          magma_trans_t trans, magma_diag_t diag,
          index_type n, const value_type* dA, index_type ldda,
          value_type* dx, index_type incx);

template<>
inline void trmv(std::shared_ptr<Context<CUDA>> context, magma_uplo_t uplo,
          magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaDouble_const_ptr dA, magma_int_t ldda,
          magmaDouble_ptr dx, magma_int_t incx)
{
    magma_dtrmv(uplo, trans, diag, n, dA, ldda, dx, incx,
        context->get_queue());
}

template<>
inline void trmv(std::shared_ptr<Context<CUDA>> context, magma_uplo_t uplo,
          magma_trans_t trans, magma_diag_t diag,
          magma_int_t n, magmaFloat_const_ptr dA, magma_int_t ldda,
          magmaFloat_ptr dx, magma_int_t incx)
{
    magma_strmv(uplo, trans, diag, n, dA, ldda, dx, incx,
        context->get_queue());
}

template <ContextType device_type, typename value_type, typename index_type>
index_type trtri(std::shared_ptr<Context<device_type>> context, magma_uplo_t uplo,
                 magma_diag_t diag, index_type n,
                 value_type* dA, index_type ldda, index_type* info);

template<>
inline magma_int_t trtri(std::shared_ptr<Context<CUDA>> context, magma_uplo_t uplo,
                  magma_diag_t diag, magma_int_t n,
                  magmaDouble_ptr dA, magma_int_t ldda, magma_int_t* info) {
    return magma_dtrtri_gpu(uplo, diag, n, dA, ldda, info);
}

template<>
inline magma_int_t trtri(std::shared_ptr<Context<CUDA>> context, magma_uplo_t uplo,
                  magma_diag_t diag, magma_int_t n,
                  magmaFloat_ptr dA, magma_int_t ldda, magma_int_t* info) {
    return magma_strtri_gpu(uplo, diag, n, dA, ldda, info);
}


template <ContextType device_type, typename value_type, typename index_type>
void trmm(std::shared_ptr<Context<device_type>> context,
         magma_side_t side, magma_uplo_t uplo,
         magma_trans_t trans, magma_diag_t diag, index_type m,
         index_type n, value_type alpha, const value_type* dA,
         index_type ldda, value_type* dB, index_type lddb);

template<>
inline void trmm(std::shared_ptr<Context<CUDA>> context,
          magma_side_t side, magma_uplo_t uplo,
          magma_trans_t trans, magma_diag_t diag, magma_int_t m,
          magma_int_t n, double alpha, magmaDouble_const_ptr dA,
          magma_int_t ldda, magmaDouble_ptr dB, magma_int_t lddb)
{
    magma_dtrmm(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB,
                lddb, context->get_queue());
}

template<>
inline void trmm(std::shared_ptr<Context<CUDA>> context,
          magma_side_t side, magma_uplo_t uplo,
          magma_trans_t trans, magma_diag_t diag, magma_int_t m,
          magma_int_t n, float alpha, magmaFloat_const_ptr dA,
          magma_int_t ldda, magmaFloat_ptr dB, magma_int_t lddb)
{
    magma_strmm(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB,
                lddb, context->get_queue());
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

template <ContextType device_type, typename value_type, typename index_type>
void gemm(std::shared_ptr<Context<device_type>>,
          magma_trans_t transA, magma_trans_t transB, index_type m,
          index_type n, magma_int_t k, value_type alpha, const value_type* dA,
          index_type ldda, const value_type* dB, index_type lddb,
          value_type beta, value_type* dC, index_type lddc);

template<>
inline void gemm(std::shared_ptr<Context<CUDA>> context,
          magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, double alpha, magmaDouble_const_ptr dA,
          magma_int_t ldda, magmaDouble_const_ptr dB, magma_int_t lddb,
          double beta, magmaDouble_ptr dC, magma_int_t lddc)
{
    magma_dgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                lddc, context->get_queue());
    cudaDeviceSynchronize();
}

template<>
inline void gemm(std::shared_ptr<Context<CUDA>> context,
          magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, float alpha, magmaFloat_const_ptr dA,
          magma_int_t ldda, magmaFloat_const_ptr dB, magma_int_t lddb,
          float beta, magmaFloat_ptr dC, magma_int_t lddc)
{
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                lddc, context->get_queue());
    cudaDeviceSynchronize();
}

inline void gemm(std::shared_ptr<Context<CUDA>> context,
          magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, magmaHalf alpha, magmaHalf_const_ptr dA,
          magma_int_t ldda, magmaHalf_const_ptr dB, magma_int_t lddb,
          magmaHalf beta, magmaHalf_ptr dC, magma_int_t lddc)
{
    magma_hgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC,
                lddc, context->get_queue());
    cudaDeviceSynchronize();
}

template<>
inline void gemm(std::shared_ptr<Context<CPU>> context,
          magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, double alpha, magmaDouble_const_ptr dA,
          magma_int_t ldda, magmaDouble_const_ptr dB, magma_int_t lddb,
          double beta, magmaDouble_ptr dC, magma_int_t lddc)
{
}

template<>
inline void gemm(std::shared_ptr<Context<CPU>> context,
          magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, float alpha, magmaFloat_const_ptr dA,
          magma_int_t ldda, magmaFloat_const_ptr dB, magma_int_t lddb,
          float beta, magmaFloat_ptr dC, magma_int_t lddc)
{
}

inline void gemm(std::shared_ptr<Context<CPU>> context,
          magma_trans_t transA, magma_trans_t transB, magma_int_t m,
          magma_int_t n, magma_int_t k, magmaHalf alpha, magmaHalf_const_ptr dA,
          magma_int_t ldda, magmaHalf_const_ptr dB, magma_int_t lddb,
          magmaHalf beta, magmaHalf_ptr dC, magma_int_t lddc)
{
}

template<ContextType device_type, typename value_type, typename index_type>
magma_int_t geqrf2(std::shared_ptr<Context<device_type>> context, index_type m,
                   index_type n, value_type* dA, index_type ldda,
                   value_type* tau, index_type* info);

template<>
inline magma_int_t geqrf2(std::shared_ptr<Context<CUDA>> context, magma_int_t m,
                   magma_int_t n, magmaDouble_ptr dA, magma_int_t ldda,
                   double* tau, magma_int_t* info)
{
    return magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
}

template<>
inline magma_int_t geqrf2(std::shared_ptr<Context<CUDA>> context, magma_int_t m,
                   magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda,
                   float* tau, magma_int_t* info)
{
    return magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
}


template<ContextType device_type, typename value_type, typename index_type>
inline value_type dot(std::shared_ptr<Context<device_type>> context, index_type n,
           const value_type* dx, index_type incx,
           const value_type* dy, index_type incy);

template<>
inline double dot(std::shared_ptr<Context<CUDA>> context, magma_int_t n,
           magmaDouble_const_ptr dx, magma_int_t incx,
           magmaDouble_const_ptr dy, magma_int_t incy)	
{
    return magma_ddot(n, dx, incx, dy, incy, context->get_queue());
}

template<>
inline float dot(std::shared_ptr<Context<CUDA>> context, magma_int_t n,
          magmaFloat_const_ptr dx, magma_int_t incx,
          magmaFloat_const_ptr dy, magma_int_t incy)	
{
    return magma_sdot(n, dx, incx, dy, incy, context->get_queue());
}


}  // end of namespace blas
}  // end of namespace rls

#endif
