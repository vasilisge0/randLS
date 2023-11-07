#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>


#include "../../core/matrix/dense/dense.hpp"
#include "../../core/memory/magma_context.hpp"
#include "../../utils/io.hpp"


#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 32


namespace rls {
namespace sketch{


template <typename index_type>
__global__ void gaussian_kernel(index_type num_rows, index_type num_cols,
                                double* mtx, index_type ld_mtx)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    thrust::minstd_rand rng;
    thrust::random::normal_distribution<double> dist(0.0f, 1.0f);
    //printf("testin===>\n");
    if ((row < num_rows) && (col < num_cols)) {
        printf("dist(rng): %lf\n", dist(rng));
        mtx[row + ld_mtx*col] = dist(rng);
        //mtx[row + ld_mtx*col] = 1.0;
        printf("mtx[%d]: %lf\n", row + ld_mtx*col, mtx[row + ld_mtx*col]);
    }
}

template <typename index_type>
__global__ void gaussian_kernel(index_type num_rows, index_type num_cols,
                                float* mtx, index_type ld_mtx)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    thrust::minstd_rand rng;
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    if ((row < num_rows) && (col < num_cols)) {
        mtx[row + ld_mtx*col] = dist(rng);
    }
}

void gaussian_sketch_impl(std::shared_ptr<matrix::Dense<CUDA, double>> mtx)
{
//    auto context = mtx->get_context();
//    auto size = mtx->get_size();
//    auto status = curandGenerateNormalDouble(
//        context->get_generator(), mtx->get_values(), size[0] * size[1], 0, 1);
//    cudaDeviceSynchronize();
//    auto num_rows = mtx->get_size()[0];
//    auto num_cols = mtx->get_size()[1];
//    int num_threads = CUDA_MAX_NUM_THREADS_PER_BLOCK_2D;
//    dim3 threads_per_block(num_threads, num_threads);
//    dim3 num_blocks((num_rows + threads_per_block.x - 1) / threads_per_block.x,
//                    (num_cols + threads_per_block.y - 1) / threads_per_block.y);
//    gaussian_kernel<<<num_blocks, threads_per_block>>>(mtx->get_size()[0],
//            mtx->get_size()[1], mtx->get_values(), mtx->get_ld());
//    cudaDeviceSynchronize();
    auto context = mtx->get_context();
    auto size = mtx->get_size();
    std::cout << "size[0]: " << size[0] << ", size[1]: " << size[1] << '\n';
    auto status = curandGenerateNormalDouble(
        context->get_generator(), mtx->get_values(), size[0] * size[1], 0, 1);
    std::cout << "status: " << status << '\n';
    //{
    //    auto queue = context->get_queue();
    //    io::write_mtx("S1.mtx", mtx->get_size()[0], mtx->get_size()[1],
    //        (double*)mtx->get_values(), mtx->get_ld(), queue);
    //    std::cout << "<double>\n";
    //}
}

void gaussian_sketch_impl(std::shared_ptr<matrix::Dense<CUDA, float>> mtx)
{
    auto context = mtx->get_context();
    auto size = mtx->get_size();
    auto status = curandGenerateNormal(context->get_generator(), mtx->get_values(), size[0] * size[1], 0, 1);
    std::cout << "status: " << status << '\n';
    //{
    //    auto queue = context->get_queue();
    //    io::write_mtx("S1.mtx", mtx->get_size()[0], mtx->get_size()[1],
    //        (float*)mtx->get_values(), mtx->get_ld(), queue);
    //    std::cout << "<float>\n";
    //}
}


}   // end of namespace sketch
}   // end of namespace rls
