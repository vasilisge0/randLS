#ifndef DETAIL_HPP
#define DETAIL_HPP


#include <memory>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


namespace rls {


#define CUDA_MAX_NUM_THREADS_PER_BLOCK 1024
#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 32


class MagmaContext {
public:    
    MagmaContext() {
        magma_init();
        cudaStreamCreate(&cuda_stream);
        cublasCreate(&cublas_handle);
        cusparseCreate(&cusparse_handle);
        magma_getdevices(&cuda_device, 1, &num_devices);
        magma_queue_create_from_cuda(
            cuda_device, cuda_stream,
            cublas_handle, cusparse_handle,
            &queue);
        curandCreateGenerator(&rand_generator,
                              CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rand_generator, time(NULL));
    }

    ~MagmaContext() {
        cudaStreamDestroy(cuda_stream);
        cublasDestroy(cublas_handle);
        cusparseDestroy(cusparse_handle);
        curandDestroyGenerator(rand_generator);
        magma_queue_destroy(queue);
        magma_finalize();
    }
    

    static std::unique_ptr<MagmaContext> create() {
        return std::make_unique<MagmaContext>();
    }

    void use_tf32_math_operations()
    {
        cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    void disable_tf32_math_operations()
    {
        cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    }

private:
    magma_queue_t queue;
    cudaStream_t cuda_stream;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    magma_device_t cuda_device;
    curandGenerator_t rand_generator;
    magma_int_t num_devices;
};


} // end of namespace rls

#endif
