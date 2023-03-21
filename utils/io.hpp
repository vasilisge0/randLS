#ifndef UTILS
#define UTILS


#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"
#include "mmio.h"


namespace rls {
namespace io {


void read_mtx_size(char* filename, magma_int_t* m, magma_int_t* n);

void read_mtx_values(char* filename, magma_int_t m, magma_int_t n, double* mtx);

void read_mtx_values(char* filename, magma_int_t m, magma_int_t n, float* mtx);

void write_mtx(char* filename, magma_int_t m, magma_int_t n, double* mtx);

void write_mtx(char* filename, magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue);

// void print_mtx(magma_int_t m, magma_int_t n, double* mtx);

void print_mtx(magma_int_t m, magma_int_t n, double* mtx, magma_int_t ld);

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue);

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, float* dmtx, magma_int_t ld, magma_queue_t queue);

void write_output(char* filename, magma_int_t num_rows, magma_int_t num_cols, magma_int_t max_iter,
    double sampling_coeff, magma_int_t sampled_rows, double t_precond, double t_solve, double t_total, double t_mm, double t_qr,
    magma_int_t iter, double relres);


}
}

#endif
