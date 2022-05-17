#ifndef BLENDNPIK_MEMORY_HPP
#define BLENDNPIK_MEMORY_HPP


#include "magma_v2.h"


namespace rls {
namespace memory {


void malloc(magmaDouble_ptr* ptr, size_t n);

void malloc(magmaFloat_ptr* ptr, size_t n);

void malloc(magmaHalf_ptr* ptr, size_t n);

void malloc_cpu(double** ptr_ptr, size_t n);

void malloc_cpu(float** ptr_ptr, size_t n);

void free(magmaDouble_ptr ptr);

void free(magmaFloat_ptr ptr);

void free(magmaHalf_ptr ptr);

void free_cpu(magmaDouble_ptr ptr);

void free_cpu(magmaFloat_ptr ptr);

void setmatrix(magma_int_t m, magma_int_t n, double* A, magma_int_t ldA,
               double* B, magma_int_t ldB, magma_queue_t queue);

void setmatrix(magma_int_t m, magma_int_t n, float* A, magma_int_t ldA,
               float* B, magma_int_t ldB, magma_queue_t queue);


}  // end of namespace memory
}  // end of namespace rls


#endif
