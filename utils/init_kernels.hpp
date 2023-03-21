#ifndef TEST_KERNELS
#define TEST_KERNELS

#include "../core/memory/magma_context.hpp"

namespace rls {
namespace utils {
// namespace {

template <typename value_type, typename index_type>
void initialize(index_type matrix_selection, index_type* num_rows,
                index_type* num_cols, value_type** mtx, value_type** dmtx,
                value_type** init_sol, value_type** sol, value_type** rhs,
                detail::magma_info& magma_config);

template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(index_type matrix_selection,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, value_type sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond);

// template <typename value_type>
void finalize(void* mtx, void* dmtx, void* init_sol,
              void* sol, void* rhs,
              detail::magma_info& magma_config);

template <typename value_type>
void finalize_with_precond(value_type* mtx, value_type* dmtx,
                           value_type* init_sol, value_type* sol,
                           value_type* rhs, value_type* precond_mtx,
                           detail::magma_info& magma_config);

// } // end of anonymous namespace
}  // namespace test

namespace utils {

template <typename value_type, typename index_type>
void initialize(std::string filename_mtx, index_type* num_rows_io,
                index_type* num_cols_io, value_type** mtx, value_type** dmtx,
                value_type** init_sol, value_type** sol, value_type** rhs,
                detail::magma_info& magma_config);

template <typename value_type, typename index_type>
void initialize(std::string filename_mtx, std::string filename_rhs, index_type* num_rows_io,
                index_type* num_cols_io, value_type** mtx, value_type** dmtx,
                value_type** init_sol, value_type** sol, value_type** rhs,
                detail::magma_info& magma_config);

template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(std::string filename_mtx,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, value_type sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond);

template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(std::string filename_mtx, std::string filename_rhs,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, double sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond);

template <typename value_type>
void finalize(value_type* mtx, value_type* dmtx, value_type* init_sol,
              value_type* sol, value_type* rhs,
              detail::magma_info& magma_config);

template <typename value_type>
void finalize_with_precond(value_type* mtx, value_type* dmtx,
                           value_type* init_sol, value_type* sol,
                           value_type* rhs, value_type* precond_mtx,
                           detail::magma_info& magma_config);       

template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond_tf32(std::string filename_mtx, std::string filename_rhs,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, value_type sampling_coeff, index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config, double* t_precond);

}
}  // namespace rls

#endif
