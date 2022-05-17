#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <string>
#include <time.h>


#include "core/detail.hpp"
#include "core/memory.hpp"
#include "cuda/kernels.hpp"
#include "utils/io.hpp"


namespace rls {
namespace test {


template <typename value_type_internal, typename value_type, typename index_type>
void allocate_iteration_data(index_type num_rows, index_type num_cols,
                             solver::iteration_data& data)
{
    if (!std::is_same<value_type_internal, value_type>::value) {
        memory::malloc(&data.u_vector, num_rows);
        memory::malloc(&data.v_vector, num_rows);
        memory::malloc(&data.w_vector, num_cols);
        memory::malloc(&data.tmp_vector, num_rows);
        memory::malloc(&data.u_vector_rp, num_rows);
        memory::malloc(&data.v_vector_rp, num_rows);
        memory::malloc(&data.w_vector_rp, num_cols);
    }
    else {
        memory::malloc(&data.u_vector, num_rows);
        memory::malloc(&data.v_vector, num_rows);
        memory::malloc(&data.w_vector, num_cols);
        memory::malloc(&data.tmp_vector, num_rows);
    }
}

template <typename value_type_internal, typename value_type>
void free_memory(solver::iteration_data& data)
{ 
    if (!std::is_same<value_type_internal, value_type>::value) {
        magma_free(data.u_vector);
        magma_free(data.v_vector);
        magma_free(data.w_vector);
        magma_free(data.tmp_vector);
        magma_free(data.u_vector_rp);
        magma_free(data.v_vector_rp);
        magma_free(data.w_vector_rp);
    }
    else {
        magma_free(data.u_vector);
        magma_free(data.v_vector);
        magma_free(data.w_vector);
        magma_free(data.tmp_vector);
    }
}


template <typename value_type_internal_precond, typename value_type_internal_solve,
    typename value_type, typename index_type>
void initialize(index_type matrix_selection, index_type* num_rows_io,
    index_type* num_cols_io, value_type** mtx, value_type** dmtx,
    value_type** init_sol, value_type** sol, value_type** rhs,
    iteration_data& data, detail::magma_info& magma_config)
{
    detail::configure_magma(magma_config);
    std::string filename_mtx;
    std::string filename_rhs;
    bool read_rhs_from_file = false;
    switch (matrix_selection) {
    case 0:
        filename_mtx = "data/lsqr_0.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
        break;
    case 1:
        filename_mtx = "data/cbcl1.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
        break;
    case 2:
        filename_mtx = "data/batch1_2_reduced.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
        break;
    case 3:
        filename_mtx = "data/hgdp_0.mtx";
        filename_rhs = "data/hgdp_0_rhs_2.mtx";
        read_rhs_from_file = true;
        break;
    case 4:
        filename_mtx = "data/hgdp_1.mtx";
        filename_rhs = "data/hgdp_1_rhs.mtx";
        read_rhs_from_file = true;
        break;
    case 5:
        filename_mtx = "data/hgdp_1_large_10e6.mtx";
        filename_rhs = "data/hgdp_1_large_10e6_b.mtx";
        read_rhs_from_file = true;
        break;    
    default:
        filename_mtx = "data/lsqr_0.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
    }
    index_type num_rows = 0;
    index_type num_cols = 0;
    rls::io::read_mtx_size((char*)filename_mtx.c_str(), &num_rows, &num_cols);
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
       cuda::solution_initialization(num_cols, *init_sol, *sol, magma_config.queue);
    }
    else {
        cuda::default_initialization(magma_config.queue, num_rows, num_cols, *dmtx,
                                 *init_sol, *sol, *rhs);
    }
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    allocate_iteration_data<value_type_internal_solve>(num_rows, num_cols, data);
}

template void initialize(magma_int_t matrix_selection, magma_int_t* num_rows,
                         magma_int_t* num_cols, double** mtx, double** d_mtx,
                         double** init_sol, double** sol, double** rhs,
                         detail::magma_info& magma_config);


template <typename value_type_internal_precond, typename value_type_internal_solver,
    typename value_type, typename index_type>
void initialize_with_precond(index_type matrix_selection,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, value_type sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond)
{
    detail::configure_magma(magma_config);
    std::string filename_mtx;
    std::string filename_rhs;
    bool read_rhs_from_file = false;
    switch (matrix_selection) {
    case 0:
        filename_mtx = "data/lsqr_0.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
        break;
    case 1:
        filename_mtx = "data/cbcl1.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
        break;
    case 2:
        filename_mtx = "data/batch1_2_reduced.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
        break;
    case 3:
        filename_mtx = "data/hgdp_0.mtx";
        filename_rhs = "data/hgdp_0_rhs_2.mtx";
        read_rhs_from_file = true;
        break;
    case 4:
        filename_mtx = "data/hgdp_1.mtx";
        filename_rhs = "data/hgdp_1_rhs.mtx";
        read_rhs_from_file = true;
        break;
    case 5:
        // option 1: works, but for sampling coefficient > 2
        // needs more memory. Check that you have the
        // correct dimensions before using.
        // filename_mtx = "data/hgdp_1_large_10e6.mtx";
        // filename_rhs = "data/hgdp_1_large_10e6_b.mtx";

        // option 2: try to make it work for sampling coefficients > 2.
        filename_mtx = "data/hgdp_1_large_10e6.mtx";
        filename_rhs = "data/hgdp_1_large_10e6_reduced_b.mtx";
        read_rhs_from_file = true;
        break;
    default:
        filename_mtx = "data/lsqr_0.mtx";
        // Uses default rhs.
        read_rhs_from_file = false;
    }
    index_type num_rows;
    index_type num_cols;
    std::cout << "matrix: " << filename_mtx.c_str() << "\n";

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

    if (read_rhs_from_file) {
        value_type* rhs_tmp = nullptr;
        memory::malloc_cpu(&rhs_tmp, num_rows);
        io::read_mtx_values((char*)filename_rhs.c_str(), num_rows, 1, rhs_tmp);
        memory::setmatrix(num_rows, 1, rhs_tmp, num_rows, *rhs, num_rows,
                     magma_config.queue);
        memory::free_cpu(rhs_tmp);
        cuda::solution_initialization(num_cols, *init_sol, *sol, magma_config.queue);
    }
    else {
        cuda::default_initialization(magma_config.queue, num_rows, num_cols, *dmtx,
                                 *init_sol, *sol, *rhs);
                                 cudaDeviceSynchronize();
    }
    
    // Generates sketch matrix.
    value_type* sketch_mtx = nullptr;
    index_type sampled_rows = (index_type)(sampling_coeff * num_cols);
    memory::malloc(&sketch_mtx, sampled_rows * num_rows);
    memory::malloc(precond_mtx, sampled_rows * num_cols);
    rls::solver::allocate_memory()
    allocate_memory(num_rows, data);
    std::cout << "sampled_rows: " << sampled_rows << '\n';
    std::cout << "num_cols: " << num_cols << '\n';
    std::cout << "Generating (sampled_rows * num_rows): "
              << sampled_rows * num_rows << ", random entries" << '\n';
    clock_t t = clock();
    auto status = curandGenerateNormalDouble(
        magma_config.rand_generator, sketch_mtx, sampled_rows * num_rows, 0, 1);
    cudaDeviceSynchronize();

    // Generates preconditioner.
    value_type* dt = nullptr;
    memory::malloc(&dt, sampled_rows * num_cols);
    cuda::generate_preconditioner<value_type_in>(
        sampled_rows, num_rows, sketch_mtx, sampled_rows, num_rows, num_cols,
        *dmtx, num_rows, *precond_mtx, sampled_rows, dt, magma_config, t_precond);
    memory::free(dt);
    memory::free(sketch_mtx);
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    *sampled_rows_io = sampled_rows;
    allocate_iteration_data<value_type_internal_solve>(num_rows, num_cols, data);
}

template void initialize_with_precond<__half>(
    magma_int_t matrix_selection, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond<float>(
    magma_int_t matrix_selection, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond<double>(
    magma_int_t matrix_selection, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);


template <typename value_type>
void finalize(value_type* mtx, value_type* dmtx, value_type* init_sol,
              value_type* sol, value_type* rhs,
              detail::magma_info& magma_config)
{
    magma_free_cpu(mtx);
    magma_free(dmtx);
    magma_free(init_sol);
    magma_free(sol);
    magma_free(rhs);
    cudaStreamDestroy(magma_config.cuda_stream);
    cublasDestroy(magma_config.cublas_handle);
    cusparseDestroy(magma_config.cusparse_handle);
    curandDestroyGenerator(magma_config.rand_generator);
    magma_queue_destroy(magma_config.queue);
    magma_finalize();
}

template void finalize(double* mtx, double* dmtx, double* init_sol, double* sol,
                       double* rhs, detail::magma_info& magma_config);


template <typename value_type>
void finalize_with_precond(value_type* mtx, value_type* dmtx,
                           value_type* init_sol, value_type* sol,
                           value_type* rhs, value_type* precond_mtx,
                           detail::magma_info& magma_config)
{
    memory::free_cpu(mtx);
    memory::free(dmtx);
    memory::free(init_sol);
    memory::free(sol);
    memory::free(rhs);
    memory::free(precond_mtx);
    cudaStreamDestroy(magma_config.cuda_stream);
    cublasDestroy(magma_config.cublas_handle);
    cusparseDestroy(magma_config.cusparse_handle);
    curandDestroyGenerator(magma_config.rand_generator);
    magma_queue_destroy(magma_config.queue);
    magma_finalize();
}

template void finalize_with_precond(double* mtx, double* dmtx, double* init_sol,
                                    double* sol, double* rhs,
                                    double* precond_mtx,
                                    detail::magma_info& magma_config);
}  // end of namespace test


namespace utils {


template <typename value_type_internal_precond, typename value_type_internal_solve,
    typename value_type, typename index_type>
void initialize(std::string filename_mtx, index_type* num_rows_io,
                index_type* num_cols_io, value_type** mtx, value_type** dmtx,
                value_type** init_sol, value_type** sol, value_type** rhs,
                detail::magma_info& magma_config)
{
    detail::configure_magma(magma_config);
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
    allocate_iteration_data<value_type_internal_solve>(num_rows, num_cols, data);
}

template void initialize(std::string filename_mtx, magma_int_t* num_rows,
                         magma_int_t* num_cols, double** mtx, double** d_mtx,
                         double** init_sol, double** sol, double** rhs,
                         detail::magma_info& magma_config);


template <typename value_type, typename index_type>
void initialize(std::string filename_mtx, std::string filename_rhs, index_type* num_rows_io,
                index_type* num_cols_io, value_type** mtx, value_type** dmtx,
                value_type** init_sol, value_type** sol, value_type** rhs,
                detail::magma_info& magma_config)
{
    detail::configure_magma(magma_config);
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
       cuda::solution_initialization(num_cols, *init_sol, *sol, magma_config.queue);
    }
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    allocate_iteration_data(num_rows, num_cols, data);
}

template void initialize(std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows,
                         magma_int_t* num_cols, double** mtx, double** d_mtx,
                         double** init_sol, double** sol, double** rhs,
                         detail::magma_info& magma_config);


template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(std::string filename_mtx,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, value_type sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond)
{
    detail::configure_magma(magma_config);
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
                                cudaDeviceSynchronize();
    
    // Generates sketch matrix.
    value_type* sketch_mtx = nullptr;
    index_type sampled_rows = (index_type)(sampling_coeff * num_cols);
    memory::malloc(&sketch_mtx, sampled_rows * num_rows);
    memory::malloc(precond_mtx, sampled_rows * num_cols);
    std::cout << "sampled_rows: " << sampled_rows << '\n';
    std::cout << "num_cols: " << num_cols << '\n';
    auto status = curandGenerateNormalDouble(
        magma_config.rand_generator, sketch_mtx, sampled_rows * num_rows, 0, 1);
    cudaDeviceSynchronize();

    // Generates preconditioner.
    value_type* dt = nullptr;
    memory::malloc(&dt, sampled_rows * num_cols);
    cuda::generate_preconditioner<value_type_in>(
        sampled_rows, num_rows, sketch_mtx, sampled_rows, num_rows, num_cols,
        *dmtx, num_rows, *precond_mtx, sampled_rows, dt, magma_config, t_precond);
    memory::free(dt);
    memory::free(sketch_mtx);
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    *sampled_rows_io = sampled_rows;
    allocate_iteration_data(num_rows, num_cols, data);
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


template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(std::string filename_mtx, std::string filename_rhs,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, value_type sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond)
{
    std::cout << "=== INITIALIZE ===" << '\n';
    detail::configure_magma(magma_config);
    index_type num_rows = 0;
    index_type num_cols = 0;
    std::cout << "filename_mtx: " << filename_mtx << '\n';
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
    cuda::solution_initialization(num_cols, *init_sol, *sol, magma_config.queue);
    
    // Generates sketch matrix.
    value_type* sketch_mtx = nullptr;
    index_type sampled_rows = (index_type)(sampling_coeff * num_cols);
    memory::malloc(&sketch_mtx, sampled_rows * num_rows);
    memory::malloc(precond_mtx, sampled_rows * num_cols);
    auto status = curandGenerateNormalDouble(
        magma_config.rand_generator, sketch_mtx, sampled_rows * num_rows, 0, 1);
    cudaDeviceSynchronize();

    // Generates preconditioner.
    value_type* dt = nullptr;
    memory::malloc(&dt, sampled_rows * num_cols);
    *t_precond = 0;
    cuda::generate_preconditioner<value_type_in>(
        sampled_rows, num_rows, sketch_mtx, sampled_rows, num_rows, num_cols,
        *dmtx, num_rows, *precond_mtx, sampled_rows, dt, magma_config, t_precond);
    memory::free(dt);
    memory::free(sketch_mtx);
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    *sampled_rows_io = sampled_rows;
    allocate_iteration_data(num_rows, num_cols, data);
}

template void initialize_with_precond<__half>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond<float>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond<double>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);


template <typename value_type>
void finalize(value_type* mtx, value_type* dmtx, value_type* init_sol,
              value_type* sol, value_type* rhs,
              detail::magma_info& magma_config)
{
    magma_free_cpu(mtx);
    magma_free(dmtx);
    magma_free(init_sol);
    magma_free(sol);
    magma_free(rhs);
    cudaStreamDestroy(magma_config.cuda_stream);
    cublasDestroy(magma_config.cublas_handle);
    cusparseDestroy(magma_config.cusparse_handle);
    curandDestroyGenerator(magma_config.rand_generator);
    magma_queue_destroy(magma_config.queue);
    magma_finalize();
}

template void finalize(double* mtx, double* dmtx, double* init_sol, double* sol,
                       double* rhs, detail::magma_info& magma_config);


template <typename value_type>
void finalize_with_precond(value_type* mtx, value_type* dmtx,
                           value_type* init_sol, value_type* sol,
                           value_type* rhs, value_type* precond_mtx,
                           detail::magma_info& magma_config)
{
    memory::free_cpu(mtx);
    memory::free(dmtx);
    memory::free(init_sol);
    memory::free(sol);
    memory::free(rhs);
    memory::free(precond_mtx);
    cudaStreamDestroy(magma_config.cuda_stream);
    cublasDestroy(magma_config.cublas_handle);
    cusparseDestroy(magma_config.cusparse_handle);
    curandDestroyGenerator(magma_config.rand_generator);
    magma_queue_destroy(magma_config.queue);
    magma_finalize();
}

template void finalize_with_precond(double* mtx, double* dmtx, double* init_sol,
                                    double* sol, double* rhs,
                                    double* precond_mtx,
                                    detail::magma_info& magma_config);

                                    

template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond_tf32(std::string filename_mtx, std::string filename_rhs,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, value_type sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond)
{
    std::cout << "=== INITIALIZE ===" << '\n';
    detail::configure_magma(magma_config);
    index_type num_rows = 0;
    index_type num_cols = 0;
    std::cout << "filename_mtx: " << filename_mtx << '\n';
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
    cuda::solution_initialization(num_cols, *init_sol, *sol, magma_config.queue);
    
    // Generates sketch matrix.
    value_type* sketch_mtx = nullptr;
    index_type sampled_rows = (index_type)(sampling_coeff * num_cols);
    memory::malloc(&sketch_mtx, sampled_rows * num_rows);
    memory::malloc(precond_mtx, sampled_rows * num_cols);
    auto status = curandGenerateNormalDouble(
        magma_config.rand_generator, sketch_mtx, sampled_rows * num_rows, 0, 1);
    cudaDeviceSynchronize();

    // Generates preconditioner.
    value_type* dt = nullptr;
    memory::malloc(&dt, sampled_rows * num_cols);
    *t_precond = 0;
    cuda::generate_preconditioner<value_type_in>(
        sampled_rows, num_rows, sketch_mtx, sampled_rows, num_rows, num_cols,
        *dmtx, num_rows, *precond_mtx, sampled_rows, dt, magma_config, t_precond);
    memory::free(dt);
    memory::free(sketch_mtx);
    *num_rows_io = num_rows;
    *num_cols_io = num_cols;
    *sampled_rows_io = sampled_rows;
    allocate_iteration_data(num_rows, num_cols, data);
}

template void initialize_with_precond_tf32<__half>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond_tf32<float>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template void initialize_with_precond_tf32<double>(
    std::string filename_mtx, std::string filename_rhs, magma_int_t* num_rows, magma_int_t* num_cols,
    double** mtx, double** d_mtx, double** init_sol, double** sol, double** rhs,
    double sampling_coeff, magma_int_t* sampled_rows_io, double** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);
}  // end of namespace utils
}  // end of namespace rls
