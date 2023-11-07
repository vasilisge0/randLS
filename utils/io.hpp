#ifndef UTILS
#define UTILS


#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../core/memory/magma_context.hpp"
#include "base_types.hpp"
#include "mmio.h"


namespace rls {
namespace io {


void read_mtx_size(char* filename, magma_int_t* m, magma_int_t* n);

void read_mtx_values(char* filename, magma_int_t m, magma_int_t n, double* mtx);

void read_mtx_values(char* filename, magma_int_t m, magma_int_t n, float* mtx);

template <typename value_type, ContextType device_type>
void read_mtx_values(std::shared_ptr<Context<device_type>> context, char* filename, dim2 size, value_type* values);

template<> void read_mtx_values(std::shared_ptr<Context<CPU>> context, char* filename, dim2 size, double* values);

template<> void read_mtx_values(std::shared_ptr<Context<CPU>> context, char* filename, dim2 size, float* values);

template<> void read_mtx_values(std::shared_ptr<Context<CUDA>> context, char* filename, dim2 size, double* values);

template<> void read_mtx_values(std::shared_ptr<Context<CUDA>> context, char* filename, dim2 size, float* values);

void write_mtx();

void write_mtx_gpu(std::shared_ptr<Context<rls::CUDA>> context, const char* filename, magma_int_t num_rows, magma_int_t num_cols,
    magma_int_t* dmtx, magma_int_t ld);

void write_mtx_gpu(std::shared_ptr<Context<rls::CUDA>> context, const char* filename, magma_int_t num_rows, magma_int_t num_cols,
    double* dmtx, magma_int_t ld);

void write_mtx_gpu(std::shared_ptr<Context<rls::CUDA>> context, const char* filename, magma_int_t nnz, __half* dmtx);

//void write_mtx_gpu(std::shared_ptr<Context<rls::CUDA>> context, const char* filename, magma_int_t num_rows, magma_int_t num_cols,
//    float* dmtx, magma_int_t ld);

void write_mtx_gpu(std::shared_ptr<Context<rls::CUDA>> context, const char* filename, magma_int_t nnz, double* dmtx);

void write_mtx_gpu(std::shared_ptr<Context<rls::CUDA>> context, const char* filename, magma_int_t nnz, float* dmtx);

//
//void write_mtx(const char* filename, magma_int_t num_rows, magma_int_t num_cols, magma_int_t* dmtx, magma_int_t ld, magma_queue_t queue);
//
//void write_mtx(const char* filename, magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue);

//void write_mtx(const char* filename, magma_int_t num_rows, magma_int_t num_cols, float* dmtx, magma_int_t ld, magma_queue_t queue);

void write_mtx_cpu(const char* filename, magma_int_t m, magma_int_t n, double* mtx, magma_int_t ld);

void write_mtx_cpu(const char* filename, magma_int_t m, magma_int_t n, float* mtx, magma_int_t ld);

void write_mtx_cpu(const char* filename, magma_int_t m, magma_int_t n, magma_int_t* mtx, magma_int_t ld);

void print_mtx(magma_int_t m, magma_int_t n, double* mtx);

void print_mtx(magma_int_t m, magma_int_t n, double* mtx, magma_int_t ld);

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue);

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, float* dmtx, magma_int_t ld, magma_queue_t queue);

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, __half* dmtx, magma_int_t ld, magma_queue_t queue);

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, int* dmtx, magma_int_t ld, magma_queue_t queue);

template<ContextType device_type>
void print_mtx_gpu(std::shared_ptr<Context<device_type>> context, magma_int_t num_rows, magma_int_t num_cols, __half* dmtx, magma_int_t ld);

void write_output(const char* filename, magma_int_t num_rows, magma_int_t num_cols, magma_int_t max_iter,
    double sampling_coeff, magma_int_t sampled_rows, double t_precond, double t_solve, double t_total,
    magma_int_t iter, double relres);

void print_nnz(magma_int_t nnz, double* mtx);

void print_nnz(magma_int_t nnz, int* mtx);

void print_nnz_gpu(magma_int_t nnz, double* dmtx, magma_queue_t queue);

void print_nnz_gpu(magma_int_t nnz, int* dmtx, magma_queue_t queue);

template <typename value_type, ContextType device_type>
void read_mtx_values(std::shared_ptr<Context<device_type>> context, char* filename, dim2 size, value_type* values, size_t ld);

template<> void read_mtx_values(std::shared_ptr<Context<CPU>> context, char* filename, dim2 size, double* values, size_t ld);

template<> void read_mtx_values(std::shared_ptr<Context<CPU>> context, char* filename, dim2 size, float* values, size_t ld);

template<> void read_mtx_values(std::shared_ptr<Context<CUDA>> context, char* filename, dim2 size, double* values, size_t ld);

template<> void read_mtx_values(std::shared_ptr<Context<CUDA>> context, char* filename, dim2 size, float* values, size_t ld);

}   // end of namespace rls
}   // end of namespace io


#endif
