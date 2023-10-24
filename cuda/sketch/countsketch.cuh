#ifndef COUNTSKETCH_CUH
#define COUNTSKETCH_CUH


#include <cuda_runtime.h>
//#include <thrust/random/linear_congruential_engine.h>
//#include <thrust/random/uniform_int_distribution.h>


#include "../../core/matrix/dense/dense.hpp"
#include "../../core/matrix/sparse/sparse.hpp"


namespace rls {
namespace sketch {


template <ContextType device_type, typename value_type, typename index_type>
void countsketch_impl(size_t nnz_per_col, std::shared_ptr<matrix::Sparse<device_type, value_type, index_type>> sketch);
//{
//
//    // curandCreateGenerator(&rand_generator, CURAND_RNG_PSEUDO_DEFAULT);
//
//    auto mtx = sketch->get_mtx();
//    auto size = mtx->get_size();
//
//    ::dim3 grid_size(size[0]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1, size[1]/CUDA_MAX_NUM_THREADS_PER_BLOCK_2D + 1);
//    ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_2D, CUDA_MAX_NUM_THREADS_PER_BLOCK_2D);
//
//    //countsketch_kernel<<<grid_size, block_size>>>(mtx->get_size(), mtx->get_values());
//}


}   // end of namespace sketch
}   // end of namespace rls


#endif
