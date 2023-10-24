#include <cuda_runtime.h>
#include <iostream>


#include "magma_lapack.h"
#include "magma_v2.h"
#include "magma_context.hpp"


#include "../../cuda/solver/lsqr_kernels.cuh"
#include "base_types.hpp"


namespace rls {
namespace memory {


template <typename value_type, ContextType device=CPU>
void zeros(dim2 size, value_type* values);

template <>
void zeros<double, CPU>(dim2 size, double* values)
{
    auto num_elems = size[0] * size[1];
    for (auto i = 0; i < num_elems; i++) {
        values[i] = 0.0;
    }
}

template <>
void zeros<float, CPU>(dim2 size, float* values)
{
    auto num_elems = size[0] * size[1];
    for (auto i = 0; i < num_elems; i++) {
        values[i] = 0.0;
    }
}

template <>
void zeros<double, CUDA>(dim2 size, double* values)
{
    auto num_elems = size[0] * size[1];
    double zero = 0.0;
    cuda::set_values(num_elems, zero, values);
}

template <>
void zeros<float, CUDA>(dim2 size, float* values)
{
    auto num_elems = size[0] * size[1];
    float zero = 0.0;
    cuda::set_values(num_elems, zero, values);
}

template <typename value_type, ContextType device=CPU>
void eye(dim2 size, value_type* values);

template <>
void eye<double, CPU>(dim2 size, double* values)
{
    auto num_elems = size[0] * size[1];
    for (auto i = 0; i < num_elems; i++) {
        values[i] = 0.0;
    }
}

template <>
void eye<float, CPU>(dim2 size, float* values)
{
    auto num_elems = size[0] * size[1];
    for (auto i = 0; i < num_elems; i++) {
        values[i] = 0.0;
    }
}

template <>
void eye<double, CUDA>(dim2 size, double* values)
{
    cuda::set_eye(size, values, size[0]);
}

template <>
void eye<float, CUDA>(dim2 size, float* values)
{
    cuda::set_eye(size, values, size[0]);
}

template <>
void zeros<__half, CUDA>(dim2 size, __half* values)
{
    cuda::set_eye(size, values, size[0]);
}

template<ContextType device_type, typename ptr_type>
void malloc(ptr_type* ptr, size_t n);

template<>
void malloc<CUDA>(magmaDouble_ptr* ptr, size_t n)
{
    magma_dmalloc(ptr, n);
}

template<>
void malloc<CUDA>(magmaFloat_ptr* ptr, size_t n)
{
    magma_smalloc(ptr, n);
}

template<>
void malloc<CUDA>(magma_int_t** ptr, size_t n)
{
    magma_imalloc(ptr, n);
}

template<>
void malloc<CUDA>(__half** ptr, size_t n)
{
    magma_malloc((magma_ptr*)ptr, n * sizeof(magmaHalf));
    //auto status = cudaMalloc((void**)ptr, n * sizeof(magmaHalf));
    //std::cout << "n: " << n << '\n';
    //std::cout << "status: " << status << '\n';
    //std::cout << "cudaGetErrorString(status): " << cudaGetErrorString(status) << '\n';
}

template<>
void malloc<CPU>(double** ptr_ptr, size_t n)
{
    magma_dmalloc_cpu(ptr_ptr, n);
}

template<>
void malloc<CPU>(float** ptr_ptr, size_t n)
{
    magma_smalloc_cpu(ptr_ptr, n);
}

template<>
void malloc<CPU>(__half** ptr_ptr, size_t n)
{
    magma_malloc_cpu((void**)ptr_ptr, n);
}

template<>
void malloc<CPU>(magma_int_t** ptr_ptr, size_t n)
{
    magma_imalloc_cpu(ptr_ptr, n);
}

template<ContextType device_type, typename ptr_type>
void free(ptr_type ptr);

template<>
void free<CUDA>(magmaDouble_ptr ptr)
{
    magma_free(ptr);
}

template<>
void free<CUDA>(magmaFloat_ptr ptr)
{
    magma_free(ptr);
}

template<>
void free<CUDA>(magmaHalf_ptr ptr)
{
    magma_free(ptr);
    //cudaFree(ptr);
}

template<>
void free<CUDA>(magma_int_t* ptr)
{
    magma_free(ptr);
}

template<>
void free<CPU>(double* ptr)
{
    magma_free_cpu(ptr);
}

template<>
void free<CPU>(float* ptr)
{
    magma_free_cpu(ptr);
}

template<>
void free<CPU>(__half* ptr)
{
    magma_free_cpu(ptr);
}

template<>
void free<CPU>(magma_int_t* ptr)
{
    magma_free_cpu(ptr);
}

void setmatrix(magma_int_t m, magma_int_t n, double* A, magma_int_t ldA,
               double* B, magma_int_t ldB, magma_queue_t& queue)
{
    magma_dsetmatrix(m, n, A, m, B, m, queue);
}

void setmatrix(magma_int_t m, magma_int_t n, float* A, magma_int_t ldA,
               float* B, magma_int_t ldB, magma_queue_t queue)
{
    magma_ssetmatrix(m, n, A, ldA, B, ldB, queue);
}


}  // end of namespace memory
}  // end of namespace rls
