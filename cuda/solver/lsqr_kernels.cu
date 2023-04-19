#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <type_traits>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"
#include "stdio.h"

#include "base_types.hpp"

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
void solution_initialization(index_type num_rows, value_type* init_sol,
                             value_type* sol, magma_queue_t queue)
{
    index_type inc = 1;
    default_solution_initialization_kernel<<<
        num_rows / CUDA_MAX_NUM_THREADS_PER_BLOCK + 1,
        CUDA_MAX_NUM_THREADS_PER_BLOCK>>>(num_rows, init_sol, sol);
    cudaDeviceSynchronize();
    blas::copy(num_rows, init_sol, inc, sol, inc, queue);
}

template void solution_initialization(magma_int_t num_rows, double* init_sol,
                                      double* sol, magma_queue_t queue);

template void solution_initialization(magma_int_t num_rows, float* init_sol,
                                      float* sol, magma_queue_t queue);


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


template <typename value_type, typename index_type>
__global__ void set_values_1d_kernel(index_type num_elems, value_type val, value_type* values)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_elems) {
        values[row] = val;
    }
}

template __global__ void set_values_1d_kernel(magma_int_t num_elems, double val, double* values);

template __global__ void set_values_1d_kernel(magma_int_t num_elems, float val, float* values);

template __global__ void set_values_1d_kernel(magma_int_t num_elems, __half val, __half* values);

template <typename value_type, typename index_type>
void set_values(index_type num_elems, value_type val, value_type* values)
{
    set_values_1d_kernel<<<num_elems/CUDA_MAX_NUM_THREADS_PER_BLOCK + 1,
        CUDA_MAX_NUM_THREADS_PER_BLOCK>>>(num_elems, val, values);
    cudaDeviceSynchronize();
}

template void set_values(magma_int_t num_elems, double val, double* values);

template void set_values(magma_int_t num_elems, float val, float* values);

template void set_values(magma_int_t num_elems, __half val, __half* values);

template <typename value_type, typename index_type>
__global__ void set_eye_2d_kernel(index_type num_rows, index_type num_cols, value_type* values, index_type ld)
{
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        if (row == col) {
            values[row + ld*col] = 1.0;
        }
        else {
            values[row + ld*col] = 0.0;
        }
    }
}

template <typename value_type, typename index_type>
void set_eye(dim2 size, value_type* values, index_type ld)
{
    ::dim3 grid_size(size[0]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1, size[1]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1);
    ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_2D, CUDA_MAX_NUM_THREADS_PER_BLOCK_2D);
    set_eye_2d_kernel<<<grid_size, block_size>>>(size[0], size[1], values, ld);
    cudaDeviceSynchronize();
}

template void set_eye(dim2 size, double* values, magma_int_t ld);

template void set_eye(dim2 size, float* values, magma_int_t ld);
                               
template void set_eye(dim2 size, __half* values, magma_int_t ld);



template <typename value_type, typename index_type>
__global__ void set_upper_triang_2d_kernel(index_type num_rows, index_type num_cols,
    value_type* values, index_type ld)
{
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols) && (row > col)) {
        values[row + ld*col] = 0.0;
    }
}

template <typename value_type, typename index_type>
void set_upper_triang(dim2 size, value_type* values, index_type ld)
{
    ::dim3 grid_size(size[0]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1, size[1]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1);
    ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_2D, CUDA_MAX_NUM_THREADS_PER_BLOCK_2D);
    set_upper_triang_2d_kernel<<<grid_size, block_size>>>(size[0], size[1], values, ld);
    cudaDeviceSynchronize();
}

template void set_upper_triang(dim2 size, double* values, magma_int_t ld);

template void set_upper_triang(dim2 size, float* values, magma_int_t ld);

template void set_upper_triang(dim2 size, __half* values, magma_int_t ld);

}  // namespace cuda
}  // namespace rls
