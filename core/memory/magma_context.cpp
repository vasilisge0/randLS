#include "magma_context.hpp"


namespace rls {


template<> Context<CUDA>::Context()
{
    type_ = CUDA;
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
    exec_ = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
}

template<> Context<CPU>::Context()
{
    type_ = CPU;
    exec_ = gko::OmpExecutor::create();
}

template<> Context<CUDA>::~Context()
{
    cudaStreamDestroy(cuda_stream);
    cublasDestroy(cublas_handle);
    cusparseDestroy(cusparse_handle);
    curandDestroyGenerator(rand_generator);
    magma_queue_destroy(queue);
    magma_finalize();
}

template<> Context<CPU>::~Context() {}


template <ContextType device_type>
cusparseHandle_t Context<device_type>::get_cusparse_handle()
{
    return cusparse_handle;
}

template <ContextType device_type>
std::unique_ptr<Context<device_type>> Context<device_type>::create()
{
    return std::unique_ptr<Context>(new Context());
}

template <ContextType device_type>
void Context<device_type>::enable_tf32_flag() { use_tf32 = true; }

template <ContextType device_type>
void Context<device_type>::disable_tf32_flag() { use_tf32 = false; }

template <ContextType device_type>
void Context<device_type>::use_tf32_math_operations()
{
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
}

template <ContextType device_type>
void Context<device_type>::disable_tf32_math_operations()
{
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
}

template <ContextType device_type>
ContextType Context<device_type>::get_type()
{
    return type_;
}

template <ContextType device_type>
magma_queue_t Context<device_type>::get_queue()
{
    return this->queue;
}

template <ContextType device_type>
curandGenerator_t Context<device_type>::get_generator()
{
    return rand_generator;
}

template <ContextType device_type>
bool Context<device_type>::is_tf32_used()
{
    return use_tf32;
}

template <ContextType device_type>
std::shared_ptr<const gko::Executor> Context<device_type>::get_executor()
{
    return exec_;
}

template class Context<rls::CUDA>;
template class Context<rls::CPU>;


}   // end of namespace rls
