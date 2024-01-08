#include <cuda_runtime.h>
#include <time.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/sort.h>
#include <ginkgo/ginkgo.hpp>


#include "../../core/matrix/dense/dense.hpp"
#include "../../core/matrix/sparse/sparse.hpp"


namespace rls {
namespace cuda {


#define CUDA_MAX_NUM_THREADS_PER_BLOCK_1D 1024
#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 32

// template <typename value_type_out, typename value_type, typename index_type>
// __host__ void countsketch(std::shared_ptr<matrix::Dense<value_type>> mtx,
//     std::shared_ptr<matrix::Dense<value_type_out>> sketch) {

//     auto sketch_size = size->get_size();

//     ::dim3 grid_size(sketch_size[1]/CUDA_MAX_NUM_THREADS_PER_BLOCK_1D + 1);
//     ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_1D);

//     // countsketch_kernel<<<grid_size, block_size>>>(mtx->get_values(), sketch->get_values());
// }

// template <typename value_type_out, typename value_type, typename index_type>
// __global__ void countsketch_generate_mtx(value_type* mtx, value_type_out* sketch) {

// }

// template <typename value_type_out, typename value_type, typename index_type>
// __global__ void countsketch_kernel(value_type* mtx, value_type_out* sketch) {


// }
//
//template <typename value_type, typename index_type>
//__global__ void countsketch_kernel(int rows, int cols, index_type* row_ptrs,
//                                   index_type* col_idxs, value_type* values)
//{
//    int row;
//    int val_int;
//    value_type val;
//    auto col = blockDim.y*blockIdx.y + threadIdx.y;
//    if (col < cols) {
//        thrust::minstd_rand rng(col*30);
//        thrust::minstd_rand rng2(col);
////rng.discard(col);
//        rng.discard(rows);
//        rng2.discard(rows);
//        thrust::uniform_int_distribution<int> dist(0, rows-1);
//        thrust::uniform_int_distribution<int> dist_vals(0, 1);
//        row = dist(rng);
//        val_int = dist_vals(rng2);
//        values[row + rows*col] = (val_int == 0) ? -1.0 : 1.0;
//        __syncthreads();
//        atomicAdd(row_ptrs + row, 1);
//
//        printf("row: %d, col: %d, cols: %d --> %lf\n", row, col, cols, val);
//    }
//}

//template <typename value_type_out, typename value_type, typename index_type>
//__global__ void countsketch_online_kernel(dim2 mtx_size, value_type* mtx,
//    dim2 sketch_size, value_type_out* sketch, value_type_out* sketched_mtx) {
//    int row;
//    auto col = blockDim.x*blockIdx.x + threadIdx.x;
//    if (col % mtx_size[1] == 0) {
//        thrust::minstd_rand rng;
//        thrust::uniform_int_distribution<int> dist(0, sketch_size[0]-1);
//        row = dist(rng);
//    }
//    __syncthreads();
//    atomicAdd(sketched_mtx + row + sketch_size[0]*col, mtx[row + mtx_size[0]*col]);
//}

//template <typename value_type, typename index_type>
//__global__ void countsketch_kernel(int k, int rows, int cols, index_type* row_ptrs,
//                                   index_type* col_idxs, value_type* values)
//{
//    int col_idx;
//    int val_int;
//    value_type val;
//    auto row = blockDim.y*blockIdx.y + threadIdx.y;
//    auto col = blockDim.x*blockIdx.x + threadIdx.x;
//
//    //printf("row: %d, rows: %d, col: %d, k: %d\n", row, rows, col, k);
//    if ((row < rows) && (col < k)) {
//        thrust::minstd_rand rng(row*30);
//        thrust::minstd_rand rng2(row);
////rng.discard(col);
//        rng.discard(cols);
//        rng2.discard(cols);
//        thrust::uniform_int_distribution<int> dist(0, cols-1);
//        thrust::uniform_int_distribution<int> dist_vals(0, 1);
//        col_idx = dist(rng);
//        val_int = dist_vals(rng2);
//        values[row + rows*col] = (val_int == 0) ? -1.0 : 1.0;
//        __syncthreads();
//        atomicAdd(row_ptrs + row, 1);
//
//        printf("row: %d, col: %d, cols: %d --> %lf/%d\n", row, col, cols, values[row + rows*col], row + rows*col);
//    }
//}

template <typename index_type>
__global__ void set_col_idxs_kernel(double seed, int k, int rows, int cols,
                                    index_type* row_ptrs,
                                    index_type* col_idxs)
{
    int col_idx;
    int val_int;
    auto row = blockDim.y*blockIdx.y + threadIdx.y;
    auto col = blockDim.x*blockIdx.x + threadIdx.x;
    if ((row < rows) && (col < k)) {
        thrust::minstd_rand rng(row + seed);
        rng.discard(30*row);   // this is the working one
        //rng.discard(row);
        thrust::uniform_int_distribution<int> dist(0, cols-1);
        val_int = dist(rng);
        //col_idxs[row_ptrs[row] + col] = dist(rng);
        col_idxs[k*(row/k) + col] = dist(rng);
        __syncthreads();
    }
}

template <typename index_type>
__global__ void set_row_ptrs_kernel(double seed, int k, int rows, index_type* row_ptrs)
{
    auto row = blockDim.y*blockIdx.y + threadIdx.y;
    if (row < rows) {
        row_ptrs[row] = row*k;
    }
}

template <typename value_type, typename index_type>
__global__ void set_values_kernel(double seed, index_type k, index_type rows, value_type* values)
{
    int col_idx;
    int val_int;
    value_type val;
    auto row = blockDim.y*blockIdx.y + threadIdx.y;
    auto col = blockDim.x*blockIdx.x + threadIdx.x;

    if ((row < rows) && (col < k)) {
        thrust::minstd_rand rng(row*30 + seed);
        rng.discard(row);
        thrust::uniform_int_distribution<int> dist_vals(0, 1);
        val_int = dist_vals(rng);
        values[row] = (val_int == 0) ? -1.0 : 1.0;
        __syncthreads();
    }
}


template <typename index_type>
__global__ void sort_col_idxs_kernel(size_t nnz_per_row, int rows, int cols,
    index_type* row_ptrs, index_type* col_idxs)
{
    auto row = blockDim.y*blockIdx.y + threadIdx.y;
    auto rptrs = thrust::device_ptr<index_type>(row_ptrs);
    auto cidxs = thrust::device_ptr<index_type>(col_idxs);
    thrust::sort(thrust::seq, cidxs + rptrs[row], cidxs + rptrs[row] + nnz_per_row);
}


template <typename value_type, typename index_type>
__global__ void set_values_kernel2(double seed, index_type k, index_type rows, index_type cols, value_type* values)
{}

}   // end of cuda namespace


namespace sketch {

template <ContextType device_type, typename value_type, typename index_type>
void countsketch_impl(size_t nnz_per_col, std::shared_ptr<matrix::Sparse<device_type, value_type, index_type>> sketch)
{
    auto size = sketch->get_size();
    double seed = time(NULL);
    // Sets values.
    {
        ::dim3 grid_size(nnz_per_col/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1, size[0]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1);
        ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_2D,
            CUDA_MAX_NUM_THREADS_PER_BLOCK_2D);
        cuda::set_values_kernel<<<grid_size, block_size>>>(seed,
            static_cast<int>(nnz_per_col), static_cast<int>(size[0]),
            sketch->get_values());
        cudaDeviceSynchronize();

        auto context = sketch->get_context();
        auto queue = context->get_queue();
    }

    // Sets row_ptrs.
    {
        ::dim3 grid_size(1, (size[0] + 1)/CUDA_MAX_NUM_THREADS_PER_BLOCK_1D + 1);
        ::dim3 block_size(1, CUDA_MAX_NUM_THREADS_PER_BLOCK_1D);
        cuda::set_row_ptrs_kernel<<<grid_size, block_size>>>(seed,
            static_cast<int>(nnz_per_col),
            static_cast<int>(size[0]) + 1, sketch->get_row_ptrs());
        cudaDeviceSynchronize();
        auto context = sketch->get_context();
        auto queue = context->get_queue();
    }

    // Sets col_idxs.
    {
        ::dim3 grid_size(nnz_per_col/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1, size[0]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1);

        ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_2D, CUDA_MAX_NUM_THREADS_PER_BLOCK_2D);
        cuda::set_col_idxs_kernel<<<grid_size, block_size>>>(seed, nnz_per_col,
            static_cast<int>(size[0]), static_cast<int>(size[1]),
            sketch->get_row_ptrs(), sketch->get_col_idxs());
        cudaDeviceSynchronize();
        auto context = sketch->get_context();
        auto queue = context->get_queue();
    }
}

template void countsketch_impl(size_t nnz_per_col, std::shared_ptr<matrix::Sparse<CUDA, double, magma_int_t>> sketch);
template void countsketch_impl(size_t nnz_per_col, std::shared_ptr<matrix::Sparse<CUDA, float, magma_int_t>> sketch);


}   // end of namespace sketch
}   // end of namespace rls
