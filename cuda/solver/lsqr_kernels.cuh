#ifndef CUDA_KERNELS
#define CUDA_KERNELS


#include "../../core/memory/magma_context.hpp"
#include "base_types.hpp"


namespace rls {
namespace cuda {


__host__ void generate_gaussian_sketch(magma_int_t num_rows,
                                       magma_int_t num_cols, double* sketch_mtx,
                                       curandGenerator_t rand_generator);

__host__ void generate_gaussian_sketch(magma_int_t num_rows,
                                       magma_int_t num_cols, float* sketch_mtx,
                                       curandGenerator_t rand_generator);

__global__ void double2half_mtx(magma_int_t size_mtx[2], double* mtx,
                                __half* mtx_rp);

template <typename value_type, typename index_type>
__global__ void default_rhs_initialization_kernel(index_type num_rows,
                                                  value_type* rhs);

template <typename value_type, typename index_type>
__global__ void default_solution_initialization_kernel(index_type num_rows,
                                                       value_type* init_sol,
                                                       value_type* sol);

template <typename value_type_internal, typename value_type,
          typename index_type, ContextType device_type>
void generate_preconditioner(index_type num_rows_sketch,
                             index_type num_cols_sketch, value_type* dsketch,
                             index_type ld_sketch, index_type num_rows_mtx,
                             index_type num_cols_mtx, value_type* dmtx,
                             index_type ld_mtx, value_type* dr_factor,
                             index_type ld_r_factor, value_type* hat_mtx,
                             std::shared_ptr<Context<device_type>> info);

template <typename value_type_internal, typename value_type,
          typename index_type, ContextType device_type>
void generate_preconditioner(index_type num_rows_sketch,
                             index_type num_cols_sketch, value_type* dsketch,
                             index_type ld_sketch, index_type num_rows_mtx,
                             index_type num_cols_mtx, value_type* dmtx,
                             index_type ld_mtx, value_type* dr_factor,
                             index_type ld_r_factor, value_type* hat_mtx,
                             std::shared_ptr<Context<device_type>> info);

template <typename value_type_internal, typename value_type,
          typename index_type, ContextType device_type>
void generate_preconditioner(index_type num_rows_sketch,
                             index_type num_cols_sketch, value_type* dsketch,
                             index_type ld_sketch, index_type num_rows_mtx,
                             index_type num_cols_mtx, value_type* dmtx,
                             index_type ld_mtx, value_type* dr_factor,
                             index_type ld_r_factor, value_type* hat_mtx,
                             std::shared_ptr<Context<device_type>> info);

void generate_preconditioner_half(magma_int_t size_sketch[2],
                                  magmaDouble_ptr sketch,
                                  magma_int_t size_mtx[2], magmaDouble_ptr mtx,
                                  magmaDouble_ptr r_factor, magma_queue_t queue,
                                  magmaDouble_ptr hat_mtx);

void generate_preconditioner_half_sm(
    magma_int_t size_sketch[2], magmaDouble_ptr sketch, magma_int_t size_mtx[2],
    magmaDouble_ptr mtx, magmaDouble_ptr r_factor, std::string filename_sketch);

void generate_preconditioner_half_to_double(
    magma_int_t size_sketch[2], magmaDouble_ptr sketch_gpu,
    magma_int_t size_mtx[2], magmaDouble_ptr mtx_gpu, magmaDouble_ptr r_factor,
    magmaDouble_ptr hat_mtx, magma_queue_t queue);

template <typename value_type, typename index_type>
void default_initialization(magma_queue_t queue, index_type num_rows,
                            index_type num_cols, value_type* mtx,
                            value_type* init_sol, value_type* sol,
                            value_type* rhs);

template <typename value_type, typename index_type>
void solution_initialization(index_type num_rows, value_type* init_sol,
                             value_type* sol, magma_queue_t queue);

template <typename value_type, typename index_type>
void set_values(index_type num_elems, value_type val, value_type* values);

template <typename value_type, typename index_type>
void set_eye(dim2 size, value_type* values, index_type ld);

template <typename value_type, typename index_type>
void set_upper_triang(dim2 size, value_type* values, index_type ld);


}  // end of namespace cuda
}  // end of namespace rls


#endif // CUDA_KERNELS
