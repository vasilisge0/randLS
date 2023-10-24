#ifndef DETAIL_HPP
#define DETAIL_HPP


#include <iostream>
#include <memory>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"
#include <ginkgo/ginkgo.hpp>


namespace rls {


#define CUDA_MAX_NUM_THREADS_PER_BLOCK 1024
#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 32

template<typename type_in>
std::shared_ptr<type_in> share(std::unique_ptr<type_in>&& in)
{
    std::shared_ptr<type_in> out = std::move(in);
    return out;
}

enum ContextType {
    UNDEFINED,
    CPU,
    CUDA
};

template <ContextType device_type>
class Context {
public:

    static std::unique_ptr<Context> create();

    void enable_tf32_flag();

    void disable_tf32_flag();

    void use_tf32_math_operations();

    void disable_tf32_math_operations();

    ContextType get_type();

    magma_queue_t get_queue();

    curandGenerator_t get_generator();

    bool is_tf32_used();

    std::shared_ptr<const gko::Executor> get_executor();

    cusparseHandle_t get_cusparse_handle();

    Context();

    ~Context();

private:

    magma_queue_t queue;
    cudaStream_t cuda_stream;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    magma_device_t cuda_device;
    curandGenerator_t rand_generator;
    magma_int_t num_devices;
    ContextType type_ = CPU;
    bool use_tf32 = false;
    std::shared_ptr<const gko::Executor> exec_;
};


} // end of namespace rls


#endif
