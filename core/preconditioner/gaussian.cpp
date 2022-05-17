#include "../blas/blas.hpp"
#include "../memory/detail.hpp"
#include "../memory/memory.hpp"
#include "../../include/base_types.hpp"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../dense/dense.hpp"
#include "gaussian.hpp"
#include "preconditioner.hpp"


namespace rls {
namespace preconditioner {


template <typename value_type_in, typename value_type,
          typename index_type>
void generate(index_type num_rows_sketch, index_type num_cols_sketch, value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx, value_type* dmtx, index_type ld_mtx,
              value_type* dr_factor, index_type ld_r_factor,
              value_type* hat_mtx, detail::magma_info& info)
{
    // Performs matrix-matrix multiplication in value_type_in precision
    // and promotes output to value_type precision.
    if (!std::is_same<value_type_in, value_type>::value) {
        value_type_in* dmtx_rp = nullptr;
        value_type_in* dsketch_rp = nullptr;
        value_type_in* dresult_rp = nullptr;
        memory::malloc(&dmtx_rp, ld_mtx * num_cols_mtx);
        memory::malloc(&dsketch_rp, ld_sketch * num_cols_sketch);
        memory::malloc(&dresult_rp, ld_r_factor * num_cols_mtx);
        cuda::demote(num_rows_mtx, num_cols_mtx, dmtx, num_rows_mtx, dmtx_rp,
                     num_rows_mtx);
        cuda::demote(num_rows_sketch, num_cols_sketch, dsketch, num_rows_sketch,
                     dsketch_rp, num_rows_sketch);
        blas::gemm(MagmaNoTrans, MagmaNoTrans, num_rows_sketch, num_cols_mtx,
                   num_rows_mtx, 1.0, dsketch_rp, num_rows_sketch, dmtx_rp,
                   num_rows_mtx, 0.0, dresult_rp, num_rows_sketch, info);
        cudaDeviceSynchronize();
        cuda::promote(num_rows_sketch, num_cols_mtx, dresult_rp,
                      num_rows_sketch, dr_factor, num_rows_sketch);
        memory::free(dmtx_rp);
        memory::free(dsketch_rp);
        memory::free(dresult_rp);
    } else {
        blas::gemm(MagmaNoTrans, MagmaNoTrans, num_rows_sketch, num_cols_mtx,
                   num_rows_mtx, 1.0, dsketch, num_rows_sketch, dmtx,
                   num_rows_mtx, 0.0, dr_factor, ld_r_factor, info);
        cudaDeviceSynchronize();
    }

    // Performs qr factorization in value_type precision.
    magma_int_t info_qr = 0;
    value_type* tau = nullptr;
    memory::malloc_cpu(&tau, num_rows_sketch);
    blas::geqrf2_gpu(num_rows_sketch, num_cols_mtx, dr_factor, ld_r_factor, tau,
                     &info_qr);
    if (info_qr == 0) {
        printf(">>> qr exited without errors\n");
    } else {
        printf(">>> error occured during qr run, exit with code: %d\n",
               info_qr);
    }
    memory::free_cpu(tau);
}

template <typename value_type_in, typename value_type,
          typename index_type>
void generate(index_type num_rows_sketch, index_type num_cols_sketch,
              value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx,
              value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
              index_type ld_r_factor, value_type* hat_mtx,
              detail::magma_info& info, double* runtime)
{
    // Performs matrix-matrix multiplication in value_type_in precision
    // and promotes output to value_type precision.
    if (!std::is_same<value_type_in, value_type>::value) {
        value_type_in* dmtx_rp = nullptr;
        value_type_in* dsketch_rp = nullptr;
        value_type_in* dresult_rp = nullptr;
        memory::malloc(&dmtx_rp, ld_mtx * num_cols_mtx);
        memory::malloc(&dsketch_rp, ld_sketch * num_cols_sketch);
        memory::malloc(&dresult_rp, ld_r_factor * num_cols_mtx);
        auto t = magma_sync_wtime(info.queue);
        cuda::demote(num_rows_mtx, num_cols_mtx, dmtx, num_rows_mtx, dmtx_rp,
                     num_rows_mtx);
        cuda::demote(num_rows_sketch, num_cols_sketch, dsketch, num_rows_sketch,
                     dsketch_rp, num_rows_sketch);
        blas::gemm(MagmaNoTrans, MagmaNoTrans, num_rows_sketch, num_cols_mtx,
                   num_rows_mtx, 1.0, dsketch_rp, num_rows_sketch, dmtx_rp,
                   num_rows_mtx, 0.0, dresult_rp, num_rows_sketch, info);
        cudaDeviceSynchronize();
        cuda::promote(num_rows_sketch, num_cols_mtx, dresult_rp,
                      num_rows_sketch, dr_factor, num_rows_sketch);
        *runtime += (magma_sync_wtime(info.queue) - t);
        memory::free(dmtx_rp);
        memory::free(dsketch_rp);
        memory::free(dresult_rp);
    } else {
        auto t = magma_sync_wtime(info.queue);
        blas::gemm(MagmaNoTrans, MagmaNoTrans, num_rows_sketch, num_cols_mtx,
                   num_rows_mtx, 1.0, dsketch, num_rows_sketch, dmtx,
                   num_rows_mtx, 0.0, dr_factor, ld_r_factor, info);
        cudaDeviceSynchronize();
        *runtime += (magma_sync_wtime(info.queue) - t);
    }

    // Performs qr factorization in value_type precision.
    magma_int_t info_qr = 0;
    value_type* tau = nullptr;
    memory::malloc_cpu(&tau, num_rows_sketch);
    auto t = magma_sync_wtime(info.queue);
    blas::geqrf2_gpu(num_rows_sketch, num_cols_mtx, dr_factor, ld_r_factor, tau,
                     &info_qr);
    *runtime += (magma_sync_wtime(info.queue) - t);
    if (info_qr == 0) {
        printf(">>> qr exited without errors\n");
    } else {
        printf(">>> error occured during qr run, exit with code: %d\n",
               info_qr);
    }
    memory::free_cpu(tau);
}


template void generate<__half, double, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, double* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    double* dmtx, magma_int_t ld_mtx, double* dr_factor,
    magma_int_t ld_r_factor, double* hat_mtx, detail::magma_info& info);

template void generate<__half, float, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, float* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    float* dmtx, magma_int_t ld_mtx, float* dr_factor, magma_int_t ld_r_factor,
    float* hat_mtx, detail::magma_info& info);

template void generate<float, double, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, double* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    double* dmtx, magma_int_t ld_mtx, double* dr_factor,
    magma_int_t ld_r_factor, double* hat_mtx, detail::magma_info& info);

template void generate<float, float, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, float* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    float* dmtx, magma_int_t ld_mtx, float* dr_factor, magma_int_t ld_r_factor,
    float* hat_mtx, detail::magma_info& info);

template void generate<double, double, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, double* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    double* dmtx, magma_int_t ld_mtx, double* dr_factor,
    magma_int_t ld_r_factor, double* hat_mtx, detail::magma_info& info);

template void generate<__half, double, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, double* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    double* dmtx, magma_int_t ld_mtx, double* dr_factor,
    magma_int_t ld_r_factor, double* hat_mtx, detail::magma_info& info,
    double* runtime);

template void generate<__half, float, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, float* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    float* dmtx, magma_int_t ld_mtx, float* dr_factor, magma_int_t ld_r_factor,
    float* hat_mtx, detail::magma_info& info, double* runtime);

template void generate<float, double, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, double* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    double* dmtx, magma_int_t ld_mtx, double* dr_factor,
    magma_int_t ld_r_factor, double* hat_mtx, detail::magma_info& info,
    double* runtime);

template void generate<float, float, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, float* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    float* dmtx, magma_int_t ld_mtx, float* dr_factor, magma_int_t ld_r_factor,
    float* hat_mtx, detail::magma_info& info, double* runtime);

template void generate<double, double, magma_int_t>(
    magma_int_t num_rows_sketch, magma_int_t num_cols_sketch, double* dsketch,
    magma_int_t ld_sketch, magma_int_t num_rows_mtx, magma_int_t num_cols_mtx,
    double* dmtx, magma_int_t ld_mtx, double* dr_factor,
    magma_int_t ld_r_factor, double* hat_mtx, detail::magma_info& info,
    double* runtime);


}  // namespace preconditioner
}  // namespace rls
