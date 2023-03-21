#include <cuda_runtime.h>
#include <curand.h>
#include <time.h>
#include <iostream>
#include <string>


#include "../core/memory/detail.hpp"
#include "../core/memory/memory.hpp"
#include "../core/preconditioner/gaussian.hpp"
#include "../core/solver/lsqr.hpp"
#include "../cuda/solver/lsqr_kernels.cuh"
#include "../include/base_types.hpp"
#include "io.hpp"


namespace rls {
namespace utils {


void finalize(void* mtx, void* dmtx, void* init_sol, void* sol, void* rhs,
              detail::magma_info& magma_config)
{
    magma_free_cpu(mtx);
    magma_free(dmtx);
    magma_free(init_sol);
    magma_free(sol);
    magma_free(rhs);
}


template <typename value_type>
void finalize_with_precond(value_type* mtx, value_type* dmtx,
                           value_type* init_sol, value_type* sol,
                           value_type* rhs, value_type* precond_mtx,
                           detail::magma_info& magma_config)
{
    // memory::free_cpu(mtx);
    memory::free(dmtx);
    memory::free(init_sol);
    memory::free(sol);
    memory::free(rhs);
    memory::free(precond_mtx);
}

template void finalize_with_precond(double* mtx, double* dmtx, double* init_sol,
                                    double* sol, double* rhs,
                                    double* precond_mtx,
                                    detail::magma_info& magma_config);

template void finalize_with_precond(float* mtx, float* dmtx, float* init_sol,
                                    float* sol, float* rhs, float* precond_mtx,
                                    detail::magma_info& magma_config);


template <typename value_type, typename index_type>
void initialize(std::string filename_mtx, index_type* num_rows_io,
                index_type* num_cols_io, value_type** mtx, value_type** dmtx,
                value_type** init_sol, value_type** sol, value_type** rhs,
                detail::magma_info& magma_config)
{
    bool read_rhs_from_file = false;
    index_type num_rows = 0;
    index_type num_cols = 0;
    io::read_mtx_size((char*)filename_mtx.c_str(), &num_rows, &num_cols);
    std::cout << "matrix: " << filename_mtx.c_str() << "\n";
    std::cout << "rows: " << num_rows << ", cols: " << num_cols << "\n";

    memory::malloc_cpu(mtx, num_rows * num_cols);
    io::read_mtx_values((char*)filename_mtx.c_str(), num_rows, num_cols, *mtx);
    memory::malloc(dmtx, num_rows * num_cols);
    memory::setmatrix(num_rows, num_cols, *mtx, num_rows, *dmtx, num_rows,
                      magma_config.queue);
    memory::malloc(sol, num_cols);
    memory::malloc(init_sol, num_cols);
    memory::malloc(rhs, num_rows);
    cuda::default_initialization(magma_config.queue, num_rows, num_cols, *dmtx,
                                 *init_sol, *sol, *rhs);
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
}

template void initialize(std::string filename_mtx, magma_int_t* num_rows,
                         magma_int_t* num_cols, double** mtx, double** d_mtx,
                         double** init_sol, double** sol, double** rhs,
                         detail::magma_info& magma_config);


// Initializes non-preconditioned LSQR.
template <typename value_type, typename index_type>
void initialize(std::string filename_mtx, std::string filename_rhs,
                index_type* num_rows_io, index_type* num_cols_io,
                value_type** mtx, value_type** dmtx, value_type** init_sol,
                value_type** sol, value_type** rhs,
                detail::magma_info& magma_config)
{
    bool read_rhs_from_file = false;
    index_type num_rows = 0;
    index_type num_cols = 0;
    io::read_mtx_size((char*)filename_mtx.c_str(), &num_rows, &num_cols);
    std::cout << "matrix: " << filename_mtx.c_str() << "\n";
    std::cout << "rows: " << num_rows << ", cols: " << num_cols << "\n";

    memory::malloc_cpu(mtx, num_rows * num_cols);
    io::read_mtx_values((char*)filename_mtx.c_str(), num_rows, num_cols, *mtx);
    memory::malloc(dmtx, num_rows * num_cols);
    memory::setmatrix(num_rows, num_cols, *mtx, num_rows, *dmtx, num_rows,
                      magma_config.queue);
    memory::malloc(sol, num_cols);
    memory::malloc(init_sol, num_cols);
    memory::malloc(rhs, num_rows);
    if (read_rhs_from_file) {
        value_type* rhs_tmp = nullptr;
        memory::malloc_cpu(&rhs_tmp, num_rows);
        io::read_mtx_values((char*)filename_rhs.c_str(), num_rows, 1, rhs_tmp);
        memory::setmatrix(num_rows, 1, rhs_tmp, num_rows, *rhs, num_rows,
                          magma_config.queue);
        memory::free_cpu(rhs_tmp);
        cuda::solution_initialization(num_cols, *init_sol, *sol,
                                      magma_config.queue);
    }
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
}

template void initialize(std::string filename_mtx, std::string filename_rhs,
                         magma_int_t* num_rows, magma_int_t* num_cols,
                         double** mtx, double** d_mtx, double** init_sol,
                         double** sol, double** rhs,
                         detail::magma_info& magma_config);


// Initializes preconditioned LSQR.
template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(
    std::string filename_mtx, index_type* num_rows_io, index_type* num_cols_io,
    value_type** mtx, value_type** dmtx, value_type** init_sol,
    value_type** sol, value_type** rhs, value_type sampling_coeff,
    index_type* sampled_rows_io, value_type** precond_mtx,
    detail::magma_info& magma_config, double* t_precond)
{
    index_type num_rows = 0;
    index_type num_cols = 0;
    io::read_mtx_size((char*)filename_mtx.c_str(), &num_rows, &num_cols);
    std::cout << "matrix: " << filename_mtx.c_str() << "\n";
    std::cout << "rows: " << num_rows << ", cols: " << num_cols << "\n";

    // Initializes matrix and rhs.
    memory::malloc_cpu(mtx, num_rows * num_cols);
    io::read_mtx_values((char*)filename_mtx.c_str(), num_rows, num_cols, *mtx);
    memory::malloc(dmtx, num_rows * num_cols);
    memory::setmatrix(num_rows, num_cols, *mtx, num_rows, *dmtx, num_rows,
                      magma_config.queue);
    memory::malloc(sol, num_cols);
    memory::malloc(init_sol, num_cols);
    memory::malloc(rhs, num_rows);
    cuda::default_initialization(magma_config.queue, num_rows, num_cols, *dmtx,
                                 *init_sol, *sol, *rhs);

    // Generates sketch matrix.
    value_type* sketch_mtx = nullptr;
    index_type sampled_rows = (index_type)(sampling_coeff * num_cols);
    memory::malloc(&sketch_mtx, sampled_rows * num_rows);
    memory::malloc(precond_mtx, sampled_rows * num_cols);
    auto status = curandGenerateNormalDouble(
        magma_config.rand_generator, sketch_mtx, sampled_rows * num_rows, 0, 1);

    // Generates preconditioner.
    value_type* dt = nullptr;
    memory::malloc(&dt, sampled_rows * num_cols);
    preconditioner::gaussian::generate<value_type_in>(
        sampled_rows, num_rows, sketch_mtx, sampled_rows, num_rows, num_cols,
        *dmtx, num_rows, *precond_mtx, sampled_rows, dt, magma_config);
    memory::free(dt);
    memory::free(sketch_mtx);
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    *sampled_rows_io = sampled_rows;
}

template void initialize_with_precond<__half>(
    std::string filename_mtx, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond<float>(
    std::string filename_mtx, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond<double>(
    std::string filename_mtx, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);


// Initialization of preconditioned LSQR, with runtime measurement.
template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(std::string filename_mtx, std::string filename_rhs,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, double sampling_coeff,
                             index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config,
                             double* t_precond, double* t_mm, double* t_qr)
{
    std::cout << "=== INITIALIZE ===" << '\n';
    index_type num_rows = 0;
    index_type num_cols = 0;
    io::read_mtx_size((char*)filename_mtx.c_str(), &num_rows, &num_cols);
    std::cout << "matrix: " << filename_mtx.c_str() << "\n";
    std::cout << "rows: " << num_rows << ", cols: " << num_cols << "\n";

    // Initializes matrix and rhs.
    memory::malloc_cpu(mtx, num_rows * num_cols);
    io::read_mtx_values((char*)filename_mtx.c_str(), num_rows, num_cols, *mtx);
    memory::malloc(dmtx, num_rows * num_cols);
    memory::setmatrix(num_rows, num_cols, *mtx, num_rows, *dmtx, num_rows,
                      magma_config.queue);
    memory::malloc(sol, num_cols);
    memory::malloc(init_sol, num_cols);
    memory::malloc(rhs, num_rows);

    value_type* rhs_tmp = nullptr;
    memory::malloc_cpu(&rhs_tmp, num_rows);
    io::read_mtx_values((char*)filename_rhs.c_str(), num_rows, 1, rhs_tmp);
    memory::setmatrix(num_rows, 1, rhs_tmp, num_rows, *rhs, num_rows,
                      magma_config.queue);
    memory::free_cpu(rhs_tmp);
    cuda::solution_initialization(num_cols, *init_sol, *sol,
                                  magma_config.queue);

    // Generates sketch matrix.
    value_type* sketch_mtx = nullptr;
    index_type sampled_rows = (index_type)(sampling_coeff * num_cols);
    memory::malloc(&sketch_mtx, sampled_rows * num_rows);
    memory::malloc(precond_mtx, sampled_rows * num_cols);

    // auto t = magma_sync_wtime(magma_config.queue);
    if (std::is_same<value_type, double>::value) {
        auto status = curandGenerateNormalDouble(magma_config.rand_generator,
                                                 (double*)sketch_mtx,
                                                 sampled_rows * num_rows, 0, 1);
        std::cout << "generate rand status: " << status << '\n';
    } else if (std::is_same<value_type, float>::value) {
        auto status = curandGenerateNormal(magma_config.rand_generator,
                                           (float*)sketch_mtx,
                                           sampled_rows * num_rows, 0, 1);
    }
    cudaDeviceSynchronize();

    // Generates preconditioner.
    auto precond_state = new preconditioner::gaussian::state<value_type_in, value_type,
        index_type>();
    precond_state->allocate(num_rows, num_cols, sampled_rows, num_rows, sampled_rows,
        sampled_rows);
    preconditioner::gaussian::generate(
        sampled_rows, num_rows, sketch_mtx, sampled_rows, num_rows, num_cols,
        *dmtx, num_rows, *precond_mtx, sampled_rows, precond_state, magma_config,
        t_precond, t_mm, t_qr);
    memory::free(sketch_mtx);
    precond_state->free();

    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    *sampled_rows_io = sampled_rows;
}

template void initialize_with_precond<__half>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows,
    magma_int_t* num_cols, double** mtx, double** d_mtx, double** init_sol,
    double** sol, double** rhs, double sampling_coeff,
    magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond, double* t_mm,
    double* t_qr);

template void initialize_with_precond<float>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows,
    magma_int_t* num_cols, double** mtx, double** d_mtx, double** init_sol,
    double** sol, double** rhs, double sampling_coeff,
    magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond, double* t_mm,
    double* t_qr);

template void initialize_with_precond<double>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows,
    magma_int_t* num_cols, double** mtx, double** d_mtx, double** init_sol,
    double** sol, double** rhs, double sampling_coeff,
    magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond, double* t_mm,
    double* t_qr);

template void initialize_with_precond<float, float, magma_int_t>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows,
    magma_int_t* num_cols, float** mtx, float** d_mtx, float** init_sol,
    float** sol, float** rhs, double sampling_coeff,
    magma_int_t* sampled_rows_io, float** precond_mtx,
    detail::magma_info& magma_config, double* t_precond, double* t_mm,
    double* t_qr);

template void initialize_with_precond<__half, float, magma_int_t>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows,
    magma_int_t* num_cols, float** mtx, float** d_mtx, float** init_sol,
    float** sol, float** rhs, double sampling_coeff,
    magma_int_t* sampled_rows_io, float** precond_mtx,
    detail::magma_info& magma_config, double* t_precond, double* t_mm,
    double* t_qr);


}  // end of namespace utils
}  // end of namespace rls
