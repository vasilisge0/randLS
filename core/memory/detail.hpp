#ifndef DETAIL_HPP
#define DETAIL_HPP


#include <cuda_runtime.h>
#include <curand.h>
#include <time.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


namespace rls {
namespace detail {


#define CUDA_MAX_NUM_THREADS_PER_BLOCK 1024
#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 32

struct magma_info {
    magma_queue_t queue;
    cudaStream_t cuda_stream;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    magma_device_t cuda_device;
    curandGenerator_t rand_generator;
    magma_int_t num_devices;
};

void configure_magma(magma_info& magma_config);

void use_tf32_math_operations(magma_info& magma_config);

void disable_tf32_math_operations(magma_info& magma_config);

}  // end of namespace detail
}  // end of namespace rls

#endif
