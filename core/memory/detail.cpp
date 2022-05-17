#include <cuda_runtime.h>
#include <curand.h>
#include <time.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "detail.hpp"


namespace rls {
namespace detail {


void configure_magma(magma_info& magma_config)
{
    magma_init();
    cudaStreamCreate(&magma_config.cuda_stream);
    cublasCreate(&magma_config.cublas_handle);
    cusparseCreate(&magma_config.cusparse_handle);
    magma_getdevices(&magma_config.cuda_device, 1, &magma_config.num_devices);
    magma_queue_create_from_cuda(
        magma_config.cuda_device, magma_config.cuda_stream,
        magma_config.cublas_handle, magma_config.cusparse_handle,
        &magma_config.queue);
    curandCreateGenerator(&magma_config.rand_generator,
                          CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(magma_config.rand_generator, time(NULL));
}

void use_tf32_math_operations(magma_info& magma_config)
{
    cublasSetMathMode(magma_config.cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
}

void disable_tf32_math_operations(magma_info& magma_config)
{
    cublasSetMathMode(magma_config.cublas_handle, CUBLAS_DEFAULT_MATH);
}

}  // namespace detail
}  // namespace rls
