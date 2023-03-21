#include <cuda_runtime.h>
#include <iostream>


#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_context.hpp"
#include "magma_lapack.h"
#include "magma_v2.h"


namespace rls {
namespace memory {


void malloc(magmaDouble_ptr* ptr, size_t n)
{
    magma_dmalloc(ptr, n);
}

void malloc(magmaFloat_ptr* ptr, size_t n) { magma_smalloc(ptr, n); }

void malloc(__half** ptr, size_t n)
{
    magma_malloc((magma_ptr*)ptr, n * sizeof(magmaHalf));
}

void malloc_cpu(double** ptr_ptr, size_t n) { magma_dmalloc_cpu(ptr_ptr, n); }

void malloc_cpu(float** ptr_ptr, size_t n) { magma_smalloc_cpu(ptr_ptr, n); }

void malloc_cpu(magma_int_t** ptr_ptr, size_t n)
{
    magma_imalloc_cpu(ptr_ptr, n);
}

void free(magmaDouble_ptr ptr) { magma_free(ptr); }

void free(magmaFloat_ptr ptr) { magma_free(ptr); }

void free(magmaHalf_ptr ptr) { magma_free(ptr); }

void free(magma_int_t* ptr) { magma_free(ptr); }

void free_cpu(magmaDouble_ptr ptr) { magma_free_cpu(ptr); }

void free_cpu(magmaFloat_ptr ptr) { magma_free_cpu(ptr); }

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
