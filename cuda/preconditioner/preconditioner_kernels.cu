#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <string>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../../core/blas/blas.hpp"
#include "../../core/memory/magma_context.hpp"
#include "../../core/memory/memory.hpp"


#define CUDA_MAX_NUM_THREADS_PER_BLOCK 1024
#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 32


namespace rls {
namespace cuda {


__host__ void generate_gaussian_sketch(magma_int_t num_rows,
                                       magma_int_t num_cols, double* sketch_mtx,
                                       curandGenerator_t rand_generator)
{
    curandGenerateNormalDouble(rand_generator, sketch_mtx, num_rows * num_cols,
                               0, 1);
    cudaDeviceSynchronize();
}

__host__ void generate_gaussian_sketch(magma_int_t num_rows,
                                       magma_int_t num_cols, float* sketch_mtx,
                                       curandGenerator_t rand_generator)
{
    curandGenerateNormal(rand_generator, sketch_mtx, num_rows * num_cols, 0, 1);
    cudaDeviceSynchronize();
}

template <typename index_type>
__global__ void demote_kernel(index_type num_rows, index_type num_cols,
                              double* mtx, index_type ld_mtx, __half* mtx_rp,
                              index_type ld_mtx_rp)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        mtx_rp[row + ld_mtx_rp * col] = __double2half(mtx[row + ld_mtx * col]);
    }
}

template <typename index_type>
__global__ void demote_kernel(index_type num_rows, index_type num_cols,
                              double* mtx, index_type ld_mtx, float* mtx_rp,
                              index_type ld_mtx_rp)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        mtx_rp[row + ld_mtx_rp * col] = (float)(mtx[row + ld_mtx * col]);
    }
}

template <typename index_type>
__global__ void demote_kernel(index_type num_rows, index_type num_cols,
                              float* mtx, index_type ld_mtx, __half* mtx_rp,
                              index_type ld_mtx_rp)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        mtx_rp[row + ld_mtx_rp * col] = __float2half(mtx[row + ld_mtx * col]);
    }
}

template <typename index_type>
__global__ void demote_kernel(index_type num_rows, index_type num_cols,
                              double* mtx, index_type ld_mtx, double* mtx_rp,
                              index_type ld_mtx_rp)
{}

template <typename index_type>
__global__ void demote_kernel(index_type num_rows, index_type num_cols,
                              float* mtx, index_type ld_mtx, float* mtx_rp,
                              index_type ld_mtx_rp)
{}

template <typename index_type>
__global__ void promote_kernel(index_type num_rows, index_type num_cols,
                               double* mtx, index_type ld_mtx, double* mtx_ip,
                               index_type ld_mtx_ip)
{}

template <typename index_type>
__global__ void promote_kernel(index_type num_rows, index_type num_cols,
                               float* mtx, index_type ld_mtx, float* mtx_ip,
                               index_type ld_mtx_ip)
{}

template <typename index_type>
__global__ void promote_kernel(index_type num_rows, index_type num_cols,
                               __half* mtx, index_type ld_mtx, double* mtx_ip,
                               index_type ld_mtx_ip)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        mtx_ip[row + ld_mtx_ip * col] =
            (double)__half2float(mtx[row + ld_mtx * col]);
    }
}

template <typename index_type>
__global__ void promote_kernel(index_type num_rows, index_type num_cols,
                               float* mtx, index_type ld_mtx, double* mtx_ip,
                               index_type ld_mtx_ip)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        mtx_ip[row + ld_mtx_ip * col] = (double)(mtx[row + ld_mtx * col]);
    }
}

template <typename index_type>
__global__ void promote_kernel(index_type num_rows, index_type num_cols,
                               __half* mtx, index_type ld_mtx, float* mtx_ip,
                               index_type ld_mtx_ip)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        mtx_ip[row + ld_mtx_ip * col] = __half2float(mtx[row + ld_mtx * col]);
    }
}

template <typename value_type_in, typename value_type, typename index_type>
__host__ void demote(index_type num_rows, index_type num_cols, value_type* mtx,
                     index_type ld_mtx, value_type_in* mtx_rp,
                     index_type ld_mtx_rp)
{
    int num_threads = CUDA_MAX_NUM_THREADS_PER_BLOCK_2D;
    dim3 threads_per_block(num_threads, num_threads);
    dim3 num_blocks((num_rows + threads_per_block.x - 1) / threads_per_block.x,
                    (num_cols + threads_per_block.y - 1) / threads_per_block.y);
    demote_kernel<<<num_blocks, threads_per_block>>>(num_rows, num_cols, mtx,
                                                     ld_mtx, mtx_rp, ld_mtx_rp);
    cudaDeviceSynchronize();
}

template <typename value_type_in, typename value_type, typename index_type>
__host__ void promote(index_type num_rows, index_type num_cols,
                      value_type_in* mtx, index_type ld_mtx, value_type* mtx_ip,
                      index_type ld_mtx_ip)
{
    index_type num_threads = CUDA_MAX_NUM_THREADS_PER_BLOCK_2D;
    dim3 threads_per_block(num_threads, num_threads);
    dim3 num_blocks((num_rows + threads_per_block.x - 1) / threads_per_block.x,
                    (num_cols + threads_per_block.y - 1) / threads_per_block.y);
    promote_kernel<<<num_blocks, threads_per_block>>>(
        num_rows, num_cols, mtx, ld_mtx, mtx_ip, ld_mtx_ip);
    cudaDeviceSynchronize();
}


template __global__ void demote_kernel(magma_int_t num_rows,
                                       magma_int_t num_cols, double* mtx,
                                       magma_int_t ld_mtx, __half* mtx_rp,
                                       magma_int_t ld_mtx_rp);

template __global__ void demote_kernel(magma_int_t num_rows,
                                       magma_int_t num_cols, float* mtx,
                                       magma_int_t ld_mtx, __half* mtx_rp,
                                       magma_int_t ld_mtx_rp);

template __global__ void demote_kernel(magma_int_t num_rows,
                                       magma_int_t num_cols, float* mtx,
                                       magma_int_t ld_mtx, float* mtx_rp,
                                       magma_int_t ld_mtx_rp);

template __global__ void demote_kernel(magma_int_t num_rows,
                                       magma_int_t num_cols, double* mtx,
                                       magma_int_t ld_mtx, double* mtx_rp,
                                       magma_int_t ld_mtx_rp);

template __global__ void promote_kernel(magma_int_t num_rows,
                                        magma_int_t num_cols, __half* mtx,
                                        magma_int_t ld_mtx, double* mtx_ip,
                                        magma_int_t ld_mtx_ip);

template __global__ void promote_kernel(magma_int_t num_rows,
                                        magma_int_t num_cols, float* mtx,
                                        magma_int_t ld_mtx, double* mtx_ip,
                                        magma_int_t ld_mtx_ip);

template __host__ void demote(magma_int_t num_rows, magma_int_t num_cols,
                              double* mtx, magma_int_t ld_mtx, __half* mtx_rp,
                              magma_int_t ld_mtx_rp);

template __host__ void demote(magma_int_t num_rows, magma_int_t num_cols,
                              float* mtx, magma_int_t ld_mtx, __half* mtx_rp,
                              magma_int_t ld_mtx_rp);

template __host__ void demote(magma_int_t num_rows, magma_int_t num_cols,
                              double* mtx, magma_int_t ld_mtx, float* mtx_rp,
                              magma_int_t ld_mtx_rp);

template __host__ void promote(magma_int_t num_rows, magma_int_t num_cols,
                               __half* mtx, magma_int_t ld_mtx, double* mtx_ip,
                               magma_int_t ld_mtx_ip);

template __host__ void promote(magma_int_t num_rows, magma_int_t num_cols,
                               __half* mtx, magma_int_t ld_mtx, float* mtx_ip,
                               magma_int_t ld_mtx_ip);

template __host__ void promote(magma_int_t num_rows, magma_int_t num_cols,
                               float* mtx, magma_int_t ld_mtx, double* mtx_ip,
                               magma_int_t ld_mtx_ip);

}  // namespace cuda
}  // namespace rls
