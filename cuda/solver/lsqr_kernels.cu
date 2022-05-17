#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <type_traits>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"
#include "stdio.h"


#include "../../core/blas/blas.hpp"


#define CUDA_MAX_NUM_THREADS_PER_BLOCK 1024
#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 32


namespace rls {
namespace cuda {


template <typename value_type, typename index_type>
__global__ void default_rhs_initialization_kernel(index_type num_rows,
                                                  value_type* rhs)
{
    if (blockIdx.x <= num_rows / CUDA_MAX_NUM_THREADS_PER_BLOCK) {
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        rhs[row] = 1.0;
    } else if (blockIdx.x == num_rows / CUDA_MAX_NUM_THREADS_PER_BLOCK + 1) {
        if (threadIdx.x < num_rows % CUDA_MAX_NUM_THREADS_PER_BLOCK) {
            auto row = blockIdx.x * blockDim.x + threadIdx.x;
            rhs[row] = 1.0;
        }
    }
}

template <typename value_type, typename index_type>
__global__ void default_solution_initialization_kernel(index_type num_rows,
                                                       value_type* init_sol,
                                                       value_type* sol)
{
    if (blockIdx.x < num_rows / CUDA_MAX_NUM_THREADS_PER_BLOCK) {
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        init_sol[row] = 0.0;
        sol[row] = 1.0;
    } else if ((blockIdx.x == num_rows / CUDA_MAX_NUM_THREADS_PER_BLOCK) &&
               (threadIdx.x < num_rows % CUDA_MAX_NUM_THREADS_PER_BLOCK)) {
        auto row = (num_rows / CUDA_MAX_NUM_THREADS_PER_BLOCK) * blockDim.x +
                   threadIdx.x;
        init_sol[row] = 0.0;
        sol[row] = 1.0;
    }
}

template __global__ void default_rhs_initialization_kernel(magma_int_t num_rows,
                                                           double* rhs);

template __global__ void default_solution_initialization_kernel(
    magma_int_t num_rows, double* init_sol, double* sol);

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

template <typename value_type, typename index_type>
void solution_initialization(index_type num_rows, value_type* init_sol, value_type* sol, magma_queue_t queue) {
    index_type inc = 1;
    default_solution_initialization_kernel<<<
        num_rows / CUDA_MAX_NUM_THREADS_PER_BLOCK + 1,
        CUDA_MAX_NUM_THREADS_PER_BLOCK>>>(num_rows, init_sol, sol);
    cudaDeviceSynchronize();
    blas::copy(num_rows, init_sol, inc, sol, inc, queue);
}

template void solution_initialization(magma_int_t num_rows, double* init_sol, double* sol, magma_queue_t queue);

template void solution_initialization(magma_int_t num_rows, float* init_sol, float* sol, magma_queue_t queue);


template <typename value_type, typename index_type>
void default_initialization(magma_queue_t queue, index_type num_rows,
                            index_type num_cols, value_type* mtx,
                            value_type* init_sol, value_type* sol,
                            value_type* rhs)
{
    index_type inc = 1;
    default_solution_initialization_kernel<<<
        num_cols / CUDA_MAX_NUM_THREADS_PER_BLOCK + 1,
        CUDA_MAX_NUM_THREADS_PER_BLOCK>>>(num_cols, init_sol, sol);
    cudaDeviceSynchronize();
    magma_dgemv(MagmaNoTrans, num_rows, num_cols, 1.0, mtx, num_rows, sol, inc,
                0.0, rhs, inc, queue);
    magma_dcopy(num_cols, init_sol, inc, sol, inc, queue);
}

template void default_initialization(magma_queue_t queue, magma_int_t num_rows,
                                     magma_int_t num_cols, double* mtx,
                                     double* init_sol, double* sol,
                                     double* rhs);


}  // namespace cuda
}  // namespace rls
