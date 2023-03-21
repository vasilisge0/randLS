#include "mmio.h"
#include "stdio.h"
#include "magma_v2.h"
#include <cuda_runtime.h>
#include <string>
#include <iostream>

#include "../core/memory/memory.hpp"

namespace rls {
namespace io {


void read_mtx_size(char* filename, magma_int_t* m, magma_int_t* n) {
    MM_typecode matcode;
    FILE* file_handle = fopen(filename, "r");
    mm_read_banner(file_handle, &matcode);
    mm_read_mtx_array_size(file_handle, m, n);
    fclose(file_handle);
}

void read_mtx_values(char* filename, magma_int_t m, magma_int_t n, double* mtx) {
    MM_typecode matcode;
    FILE* file_handle = fopen(filename, "r");
    mm_read_banner(file_handle, &matcode);
    mm_read_mtx_array_size(file_handle, &m, &n);
    magma_int_t size[2] = {m, n};
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fscanf(file_handle, "%lf ", &mtx[i * n + j]);
        }
    }
    fclose(file_handle);
}

void read_mtx_values(char* filename, magma_int_t m, magma_int_t n, float* mtx) {
    MM_typecode matcode;
    FILE* file_handle = fopen(filename, "r");
    mm_read_banner(file_handle, &matcode);
    mm_read_mtx_array_size(file_handle, &m, &n);
    magma_int_t size[2] = {m, n};
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fscanf(file_handle, "%f ", &mtx[i * n + j]);
        }
    }
    fclose(file_handle);
}
void write_mtx(char* filename, magma_int_t m, magma_int_t n, double* mtx) {
    MM_typecode matcode;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_real(&matcode);
    FILE* file_handle = fopen(filename, "w");
    mm_write_banner(file_handle, matcode);
    mm_write_mtx_array_size(file_handle, m, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            fprintf(file_handle, "%lf\n", mtx[i + j * m]);
        }
    }
}

void write_mtx(char* filename, magma_int_t m, magma_int_t n, float* mtx) {
    MM_typecode matcode;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_real(&matcode);
    FILE* file_handle = fopen(filename, "w");
    mm_write_banner(file_handle, matcode);
    mm_write_mtx_array_size(file_handle, m, n);
    int count = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            count += 1;
            fprintf(file_handle, "%lf\n", mtx[i + j * m]);
        }
    }
}

void print_mtx(magma_int_t m, magma_int_t n, double* mtx) {
    for (auto i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%lf ", mtx[i + j * m]);
        }
        printf("\n");
    }
}

void print_mtx(magma_int_t m, magma_int_t n, double* mtx, magma_int_t ld) {
    for (auto i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%lf ", mtx[i + j * ld]);
        }
        printf("\n");
    }
}

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue)
{  
    double* t;
    magma_malloc_cpu((void**)&t, num_rows * num_cols * sizeof(double));
    magma_dgetmatrix(num_rows, num_cols, dmtx, ld,
             t, num_rows, queue);
    print_mtx(num_rows, num_cols, t, num_rows);
    magma_free_cpu(t);
}

void write_mtx(char* filename, magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue) {
    double* t;
    auto err = magma_malloc_cpu((void**)&t, num_rows * num_cols * sizeof(double));
    magma_dgetmatrix(num_rows, num_cols, dmtx, ld,
              t, num_rows, queue);
    write_mtx(filename, num_rows, num_cols, t);
    magma_free_cpu(t);
}

void write_mtx(char* filename, magma_int_t num_rows, magma_int_t num_cols, float* dmtx, magma_int_t ld, magma_queue_t queue) {
    float* t;
    magma_malloc_cpu((void**)&t, num_rows * num_cols * sizeof(float));
    magma_sgetmatrix(num_rows, num_cols, dmtx, ld,
              t, num_rows, queue);
    write_mtx(filename, num_rows, num_cols, t);
    magma_free_cpu(t);
}

void write_output(char* filename, magma_int_t num_rows, magma_int_t num_cols, magma_int_t max_iter,
    double sampling_coeff, magma_int_t sampled_rows, double t_precond, double t_solve, double t_total,
    magma_int_t iter, double relres) {
    FILE* file_handle = fopen(filename, "w");
    fprintf(file_handle, "%d\n", num_rows);
    fprintf(file_handle, "%d\n", num_cols);
    fprintf(file_handle, "%d\n", max_iter);
    fprintf(file_handle, "%lf\n", sampling_coeff);
    fprintf(file_handle, "%d\n", sampled_rows);
    fprintf(file_handle, "%lf\n", t_precond);
    fprintf(file_handle, "%lf\n", t_solve);
    fprintf(file_handle, "%lf\n", t_total);
    fprintf(file_handle, "%d\n", iter);
    fprintf(file_handle, "%lf\n", relres);
    fclose(file_handle);
}

void write_output(char* filename, magma_int_t num_rows, magma_int_t num_cols, magma_int_t max_iter,
    double sampling_coeff, magma_int_t sampled_rows, double t_precond, double t_solve, double t_total,
    double t_mm, double t_qr, magma_int_t iter, double relres) {
    FILE* file_handle = fopen(filename, "w");
    fprintf(file_handle, "%d\n", num_rows);
    fprintf(file_handle, "%d\n", num_cols);
    fprintf(file_handle, "%d\n", max_iter);
    fprintf(file_handle, "%lf\n", sampling_coeff);
    fprintf(file_handle, "%d\n", sampled_rows);
    fprintf(file_handle, "%lf\n", t_precond);
    fprintf(file_handle, "%lf\n", t_solve);
    fprintf(file_handle, "%lf\n", t_total);
    fprintf(file_handle, "%d\n", iter);
    fprintf(file_handle, "%lf\n", relres);
    fprintf(file_handle, "%lf\n", t_mm);
    fprintf(file_handle, "%lf\n", t_qr);
    fclose(file_handle);
}


}
}
