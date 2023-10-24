#include <iostream>
#include <string>
#include <iomanip>
#include <cuda_runtime.h>
#include "stdio.h"
#include "magma_v2.h"
#include "mmio.h"


#include "../core/memory/magma_context.hpp"
#include "../core/memory/memory.hpp"
#include "../core/matrix/dense/dense.hpp"
#include "../core/matrix/dense/dense.cpp"
#include "base_types.hpp"


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
    std::cout << "m: " << m << "\n";
    std::cout << "n: " << n << "\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fscanf(file_handle, "%lf ", &mtx[i * n + j]);
        }
    }
    fclose(file_handle);
}

template <typename value_type, ContextType device_type>
void read_mtx_values(std::shared_ptr<Context<device_type>> context, char* filename, dim2 size, value_type* values);

template<> void read_mtx_values<double, CPU>(std::shared_ptr<Context<CPU>> context, char* filename, dim2 size, double* values) {
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
    std::shared_ptr<Context<CPU>> context_cpu = Context<CPU>::create();
    auto tmp = matrix::Dense<CPU, double>::create(context_cpu, size);
    auto queue = context->get_queue();
    io::read_mtx_values<double, CPU>(context_cpu, filename_mtx, size,
       tmp->get_values());
    memory::setmatrix(size[0], size[1], tmp->get_values(),
       size[0], values, size[0], queue);
}

template <>
void read_mtx_values<float, CUDA>(std::shared_ptr<Context<CUDA>> context, char* filename_mtx, dim2 size, float* values) {
    std::shared_ptr<Context<CPU>> context_cpu = Context<CPU>::create();
    //auto tmp = matrix::Dense<CPU, float>::create(context_cpu, size);
    //auto queue = context->get_queue();
    //io::read_mtx_values<float, CPU>(context_cpu, filename_mtx, size,
    //   tmp->get_values());
    //memory::setmatrix(size[0], size[1], tmp->get_values(),
    //   size[0], values, size[0], queue);
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

void write_mtx() {
    std::cout << "TEST\n";
}

void write_mtx(const char* filename, magma_int_t m, magma_int_t n, double* mtx) {
    MM_typecode matcode;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_real(&matcode);
    FILE* file_handle = fopen(filename, "w");
    mm_write_banner(file_handle, matcode);
    mm_write_mtx_array_size(file_handle, m, n);
    //std::cout << "m: " << m << ", n: " << n << '\n';
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            //std::cout << mtx[i + j * m] << '\n';
            //fprintf(file_handle, "%lf\n", mtx[i + j * m]);
            fprintf(file_handle, "%1.16e\n", mtx[i + j * m]);
        }
    }
}

void write_mtx(const char* filename, magma_int_t m, magma_int_t n, float* mtx) {
    std::cout << "in write_mtx\n";
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
            //printf("%lf ", mtx[i + j * m]);
            printf("%lf ", mtx[i + j * m]);
        }
        printf("\n");
    }
}

void print_mtx(magma_int_t m, magma_int_t n, double* mtx, magma_int_t ld) {
    //printf("here\n");
    for (auto i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.15e ", mtx[i + j * ld]);
            //printf("%lf ", mtx[i + j * ld]);
            //std::cout << std::setprecision(16) << mtx[i + j * ld];
            //printf("%lf(%d) ", mtx[i + j * ld], i + j * ld);
            //printf("%1.16e ", mtx[i + j * m]);
        }
        printf("\n");
    }
}

void print_mtx(magma_int_t m, magma_int_t n, float* mtx, magma_int_t ld) {
    //printf("here\n");
    for (auto i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", mtx[i + j * ld]);
            //printf("%lf(%d) ", mtx[i + j * ld], i + j * ld);
            //printf("%1.16e ", mtx[i + j * m]);
        }
        printf("\n");
    }
}

void print_mtx(magma_int_t m, magma_int_t n, int* mtx, magma_int_t ld) {
    for (auto i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", mtx[i + j * ld]);
            //printf("%1.16e ", mtx[i + j * m]);
        }
        printf("\n");
    }
}

void print_nnz(magma_int_t nnz, double* mtx) {
    for (auto i = 0; i < nnz; ++i) {
        printf("%lf(%d) \n", mtx[i], i);
    }
    //printf("\n");
}

void print_nnz(magma_int_t nnz, int* mtx) {
    for (auto i = 0; i < nnz; ++i) {
        printf("%d(%d) / nnz: %d\n", mtx[i], i, nnz);
    }
    //printf("\n");
}

void print_nnz_gpu(magma_int_t nnz, double* dmtx, magma_queue_t queue)
{
    double* t;
    magma_malloc_cpu((void**)&t, nnz * sizeof(double));
    //t[0] = 6.1;
    //t[1] = 6.1;
    //t[2] = 6.1;
    //t[3] = 6.1;
    //t[4] = 6.1;
    //t[5] = 6.1;
    //t[6] = 6.1;
    //t[7] = 6.1;
    //t[8] = 6.1;
    //t[9] = 6.1;
    cudaDeviceSynchronize();
    magma_dgetmatrix(nnz, 1, dmtx, nnz, t, nnz, queue); // (!!) problem here
    //cudaMemcpy(t, dmtx, 1, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //std::cout << "nnz: " << nnz << "\n";
    //printf("t[0]: %1.2e\n", t[0]);
    //printf("t[1]: %1.2e\n", t[1]);
    //printf("t[2]: %1.2e\n", t[2]);
    //printf("t[3]: %1.2e\n", t[3]);
    //printf("t[4]: %1.2e\n", t[4]);
    //printf("t[5]: %1.2e\n", t[5]);
    //printf("t[6]: %1.2e\n", t[6]);
    //printf("t[7]: %1.2e\n", t[7]);
    //printf("t[8]: %1.2e\n", t[8]);
    //printf("t[9]: %1.2e\n", t[9]);
    //printf("%lf\n", t[0]);
    print_nnz(nnz, t);
    magma_free_cpu(t);
}

void print_nnz_gpu(magma_int_t nnz, int* dmtx, magma_queue_t queue)
{
    int* t;
    magma_malloc_cpu((void**)&t, nnz * sizeof(int));
    //t[0] = 6.1;
    //t[1] = 6.1;
    //t[2] = 6.1;
    //t[3] = 6.1;
    //t[4] = 6.1;
    //t[5] = 6.1;
    //t[6] = 6.1;
    //t[7] = 6.1;
    //t[8] = 6.1;
    //t[9] = 6.1;
    cudaDeviceSynchronize();
    magma_igetmatrix(nnz, 1, dmtx, nnz, t, nnz, queue); // (!!) problem here
    //cudaMemcpy(t, dmtx, 1, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //std::cout << "nnz: " << nnz << "\n";
    //printf("t[0]: %1.2e\n", t[0]);
    //printf("t[1]: %1.2e\n", t[1]);
    //printf("t[2]: %1.2e\n", t[2]);
    //printf("t[3]: %1.2e\n", t[3]);
    //printf("t[4]: %1.2e\n", t[4]);
    //printf("t[5]: %1.2e\n", t[5]);
    //printf("t[6]: %1.2e\n", t[6]);
    //printf("t[7]: %1.2e\n", t[7]);
    //printf("t[8]: %1.2e\n", t[8]);
    //printf("t[9]: %1.2e\n", t[9]);
    //printf("%lf\n", t[0]);
    print_nnz(nnz, t);
    magma_free_cpu(t);
}


template<ContextType device_type>
void print_mtx_gpu(std::shared_ptr<Context<device_type>> context, magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld)
{
    double* t;
    auto queue = context->get_queue();
    magma_malloc_cpu((void**)&t, ld * num_cols * sizeof(double));
    magma_dgetmatrix(num_rows, num_cols, dmtx, ld,
             t, ld, queue);
    print_mtx(num_rows, num_cols, t, ld);
    magma_free_cpu(t);
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

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, float* dmtx,
    magma_int_t ld, magma_queue_t queue)
{
    float* t;
    magma_malloc_cpu((void**)&t, ld * num_cols * sizeof(float));
    magma_sgetmatrix(num_rows, num_cols, dmtx, ld,
             t, ld, queue);
    print_mtx(num_rows, num_cols, t, ld);
    magma_free_cpu(t);
}

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, __half* dmtx,
    magma_int_t ld, magma_queue_t queue)
{
    __half* t;
    magma_malloc_cpu((void**)&t, ld * num_cols * sizeof(__half));
    //magma_sgetmatrix(num_rows, num_cols, dmtx, ld,
    //         t, ld, queue);
    //print_mtx(num_rows, num_cols, t, ld);
    magma_free_cpu(t);
}

void print_mtx_gpu(magma_int_t num_rows, magma_int_t num_cols, int* dmtx,
    magma_int_t ld, magma_queue_t queue)
{
    int* t;
    magma_malloc_cpu((void**)&t, ld * num_cols * sizeof(int));
    magma_igetmatrix(num_rows, num_cols, dmtx, ld,
             t, ld, queue);
    print_mtx(num_rows, num_cols, t, ld);
    magma_free_cpu(t);
}

void write_mtx(const char* filename, magma_int_t num_rows, magma_int_t num_cols, double* dmtx, magma_int_t ld, magma_queue_t queue) {
    double* t;
    auto err = magma_malloc_cpu((void**)&t, num_rows * num_cols * sizeof(double));
    magma_dgetmatrix(num_rows, num_cols, dmtx, ld,
                     t, num_rows, queue);
    write_mtx(filename, num_rows, num_cols, t);
    magma_free_cpu(t);
}

void write_mtx(const char* filename, magma_int_t num_rows, magma_int_t num_cols, float* dmtx, magma_int_t ld, magma_queue_t queue) {
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


}   // end of namespace io
}   // end of namespace rls


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
