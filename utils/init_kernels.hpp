#ifndef TEST_KERNELS
#define TEST_KERNELS


#include "../core/memory/detail.hpp"


namespace rls {
namespace utils {


template <typename value_type, typename index_type>
void initialize(std::string filename_mtx, index_type* num_rows_io,
                index_type* num_cols_io, value_type** mtx, value_type** dmtx,
                value_type** init_sol, value_type** sol, value_type** rhs,
                detail::magma_info& magma_config);

template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(
    std::string filename_mtx, index_type* num_rows_io, index_type* num_cols_io,
    value_type** mtx, value_type** dmtx, value_type** init_sol,
    value_type** sol, value_type** rhs, value_type sampling_coeff,
    index_type* sampled_rows_io, value_type** precond_mtx,
    detail::magma_info& magma_config, double* t_precond);

template <typename value_type_in, typename value_type, typename index_type>
void initialize_with_precond(std::string filename_mtx, std::string filename_rhs,
                             index_type* num_rows_io, index_type* num_cols_io,
                             value_type** mtx, value_type** dmtx,
                             value_type** init_sol, value_type** sol,
                             value_type** rhs, double sampling_coeff,
                             index_type* sampled_rows_io,
                             value_type** precond_mtx,
                             detail::magma_info& magma_config,
                             double* t_precond, double* t_mm, double* t_qr);

template <typename value_type>
void finalize(value_type* mtx, value_type* dmtx, value_type* init_sol,
              value_type* sol, value_type* rhs,
              detail::magma_info& magma_config);

template <typename value_type>
void finalize_with_precond(value_type* mtx, value_type* dmtx,
                           value_type* init_sol, value_type* sol,
                           value_type* rhs, value_type* precond_mtx,
                           detail::magma_info& magma_config);


}  // namespace utils
}  // namespace rls


#endif
