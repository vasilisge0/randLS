#ifndef RLS_MEMORY_HPP
#define RLS_MEMORY_HPP


#include "magma_v2.h"
#include "base_types.hpp"
#include "magma_context.hpp"


namespace rls {
namespace memory {

template <typename value_type, ContextType device=CPU>
void zeros(dim2 size, value_type* values);

template <typename value_type, ContextType device=CUDA>
void eye(dim2 size, value_type* values);

void malloc(magmaDouble_ptr* ptr, size_t n);

void malloc(magmaFloat_ptr* ptr, size_t n);

void malloc(magmaHalf_ptr* ptr, size_t n);

void malloc(magma_int_t** ptr, size_t n);

void malloc_cpu(double** ptr_ptr, size_t n);

void malloc_cpu(float** ptr_ptr, size_t n);

void malloc_cpu(magma_int_t** ptr_ptr, size_t n);

void free(magmaDouble_ptr ptr);

void free(magmaFloat_ptr ptr);

void free(magmaHalf_ptr ptr);

void free(magma_int_t* ptr);

void free_cpu(double* ptr);

void free_cpu(float* ptr);

void free_cpu(magma_int_t* ptr);

void setmatrix(magma_int_t m, magma_int_t n, double* A, magma_int_t ldA,
               double* B, magma_int_t ldB, magma_queue_t& queue);

void setmatrix(magma_int_t m, magma_int_t n, float* A, magma_int_t ldA,
               float* B, magma_int_t ldB, magma_queue_t queue);


}  // end of namespace memory
}  // end of namespace rls


#endif
