#include <cuda_runtime.h>
#include "../../core/dense/dense.hpp"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

namespace rls {
namespace cuda {

#define CUDA_MAX_NUM_THREADS_PER_BLOCK 1024
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

template <typename value_type_out, typename value_type, typename index_type, ContextType device_type=CUDA>
__host__ void countsketch_online(std::shared_ptr<matrix::Dense<value_type, device_type>> mtx,
    std::shared_ptr<matrix::Dense<value_type_out, device_type>> sketch,
    std::shared_ptr<matrix::Dense<value_type_out, device_type>> sketched_mtx) {

    // curandCreateGenerator(&rand_generator, CURAND_RNG_PSEUDO_DEFAULT);

    auto size = mtx->get_size();

    ::dim3 grid_size(size[0]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1, size[1]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1);
    ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_2D, CUDA_MAX_NUM_THREADS_PER_BLOCK_2D);

    countsketch_online_kernel<<<grid_size, block_size>>>(mtx->get_size(), mtx->get_values(),
        sketch->get_size(), sketch->get_values(), sketched_mtx->get_values());
}

template <typename value_type_out, typename value_type, typename index_type>
__global__ void countsketch_online_kernel(dim2 mtx_size, value_type* mtx,
    dim2 sketch_size, value_type_out* sketch, value_type_out* sketched_mtx) {
    int row;
    auto col = blockDim.x*blockIdx.x + threadIdx.x;
    if (col % mtx_size[1] == 0) {
        thrust::minstd_rand rng;
        thrust::uniform_int_distribution<int> dist(0, sketch_size[0]-1);
        row = dist(rng);
    }
    __syncthreads();
    atomicAdd(sketched_mtx + row + sketch_size[0]*col, mtx[row + mtx_size[0]*col]);
}

}   // end of cuda namespace
}   // end of rls namespace
