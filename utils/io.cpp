#include "mmio.h"
#include "stdio.h"
#include "magma_v2.h"
#include <cuda_runtime.h>
#include <string>

#include "../core/memory/magma_context.hpp"
#include "../core/dense/dense.hpp"
#include "base_types.hpp"

#include <iostream>

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

template <>
void read_mtx_values<double, CPU>(std::shared_ptr<Context<CPU>> context, char* filename, dim2 size, double* values) {
    MM_typecode matcode;
    auto m = size[0];
    auto n = size[1];
    FILE* file_handle = fopen(filename, "r");
    mm_read_banner(file_handle, &matcode);
    mm_read_mtx_array_size(file_handle, &size[0], &size[1]);
    for (int i = 0; i < m; ++i) {
       for (int j = 0; j < n; ++j) {
           fscanf(file_handle, "%lf ", &values[i * n + j]);    // this seems wrong
       }
    }
    fclose(file_handle);
}

template <>
void read_mtx_values<float, CPU>(std::shared_ptr<Context<CPU>> context, char* filename, dim2 size, float* values) {
    MM_typecode matcode;
    auto m = size[0];
    auto n = size[1];
    FILE* file_handle = fopen(filename, "r");
    mm_read_banner(file_handle, &matcode);
    mm_read_mtx_array_size(file_handle, &size[0], &size[1]);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fscanf(file_handle, "%lf ", &values[i * n + j]);    // this seems wrong
        }
    }
    fclose(file_handle);
}

template <>
void read_mtx_values<double, CUDA>(std::shared_ptr<Context<CUDA>> context, char* filename_mtx, dim2 size, double* values) {
    std::shared_ptr<Context<CPU>> context_cpu = Context<CPU>::create(CPU);
    auto tmp = matrix::Dense<double, CPU>::create(context_cpu, size);
    auto queue = context->get_queue();
    io::read_mtx_values<double, CPU>(context_cpu, filename_mtx, size,
       tmp->get_values());
    memory::setmatrix(size[0], size[1], tmp->get_values(),
       size[0], values, size[0], queue);
}

template <>
void read_mtx_values<float, CUDA>(std::shared_ptr<Context<CUDA>> context, char* filename_mtx, dim2 size, float* values) {
    std::shared_ptr<Context<CPU>> context_cpu = Context<CPU>::create(CPU);
    auto tmp = matrix::Dense<float, CPU>::create(context_cpu, size);
    auto queue = context->get_queue();
    io::read_mtx_values<float, CPU>(context_cpu, filename_mtx, size,
       tmp->get_values());
    memory::setmatrix(size[0], size[1], tmp->get_values(),
       size[0], values, size[0], queue);
}

//template <>
//void read_mtx_values<float, CUDA>(std::shared_ptr<Context> context, char* filename_mtx, dim2 size, float* values) {
//    std::shared_ptr<Context> context_cpu = Context::create(CPU);
//    auto tmp = matrix::Dense<float>::create(context_cpu, size);
//    auto queue = context->get_queue();
//    io::read_mtx_values<float, CPU>(context, filename_mtx, size,
//        tmp->get_values());
//    memory::setmatrix(size[0], size[1], tmp->get_values(),
//        size[0], values, size[0], queue);
//}

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
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            // fprintf(file_handle, "%lf\n", mtx[i + j * m]);
            fprintf(file_handle, "%1.16e\n", mtx[i + j * m]);
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
    printf("here\n");
    std::cout << "m: " << m << ", n: " << n << '\n';
    for (auto i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%lf ", mtx[i + j * ld]);
            // printf("%1.16e ", mtx[i + j * m]);
        }
        printf("\n");
    }
}

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, double* dmtx,
    magma_int_t ld, magma_queue_t queue)
{
    double* t;
    magma_malloc_cpu((void**)&t, ld * num_cols * sizeof(double));
    magma_dgetmatrix(num_rows, num_cols, dmtx, ld,
             t, ld, queue);
    print_mtx(num_rows, num_cols, t, ld);
    magma_free_cpu(t);
}

void write_mtx(char* filename, magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue) {
    double* t;
    auto err = magma_malloc_cpu((void**)&t, num_rows * num_cols * sizeof(double));
    printf("err: %d\n", err);
    printf("num_rows: %d, num_cols: %d\n", num_rows, num_cols);
    printf("ld: %d\n", ld);
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

void write_output(const char* filename, magma_int_t num_rows, magma_int_t num_cols, magma_int_t max_iter,
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


}
}

// void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, float* dmtx, magma_int_t ld, magma_queue_t queue)
// {  
//     float* t;
//     magma_malloc_cpu((void**)&t, num_rows * num_cols * sizeof(float));
//     magma_sgetmatrix(num_rows, num_cols, dmtx, ld,
//               t, num_rows, queue);
//     print_mtx(num_rows, num_cols, t, ld);          
//     magma_free_cpu(t);
// }

// {
//         value_type* t;
//         magma_malloc_cpu((void**)&t, num_rows*sizeof(double));
//         getmatrix(num_cols, 1, v_vector, num_cols,
//                 t, num_cols, queue);
//         std::cout << "v_vector: " << '\n';
//         std::cout << t[0] << '\n';
//         std::cout << t[1] << '\n';
//         std::cout << t[2] << '\n';
//         std::cout << t[3] << '\n';
//         std::cout << t[4] << '\n';
//         // std::cout << t[5] << '\n';
//         // std::cout << t[6] << '\n';
//         // std::cout << t[7] << '\n';
//         // std::cout << t[8] << '\n';
//         // std::cout << t[9] << '\n';
//         magma_free_cpu(t);
//     }


 // for (auto col = 0; col < num_cols; col++) {
    //     for (auto row = 0; row < sampled_rows; row++) {
    //         if (row >= col) {
    //             t[row + col * sampled_rows] = 1.0;
    //             std::cout << "row: " << row << ", col: " << col << ", row + col * sampled_rows: " << row + col * sampled_rows << '\n';
    //         } else {
    //             t[row + col * sampled_rows] = 0.0;
    //         }
    //     }
    // }
    // magma_dsetmatrix(sampled_rows, num_cols, t, sampled_rows, *precond_mtx,
    //                  sampled_rows, magma_config.queue);

    // magma_dgetmatrix(sampled_rows, num_cols, dt, sampled_rows,
    //                  t, sampled_rows, magma_config.queue);
    // std::cout << "hat matrix\n";
    // print_mtx(sampled_rows, num_cols, t);
    // std::cout << '\n';
