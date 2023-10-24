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

template <ContextType device_type, typename ptr_type>
void malloc(ptr_type* ptr, size_t n);

template<typename value_type>
void test_malloc(value_type t);

template<ContextType device_type, typename ptr_type>
void free(ptr_type ptr);

void setmatrix(magma_int_t m, magma_int_t n, double* A, magma_int_t ldA,
               double* B, magma_int_t ldB, magma_queue_t& queue);

void setmatrix(magma_int_t m, magma_int_t n, float* A, magma_int_t ldA,
               float* B, magma_int_t ldB, magma_queue_t queue);


}  // end of namespace memory
}  // end of namespace rls


#endif
