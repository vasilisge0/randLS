#include "../memory/detail.hpp"


namespace rls {
namespace preconditioner {
namespace gaussian {

template <typename value_type_internal, typename value_type,
          typename index_type>
struct state{ 
    value_type_internal* dmtx_rp = nullptr;
    value_type_internal* dsketch_rp = nullptr;
    value_type_internal* dresult_rp = nullptr;
    value_type* tau = nullptr;

    void allocate(index_type ld_mtx, index_type num_cols_mtx, 
        index_type num_rows_sketch, index_type num_cols_sketch, index_type ld_sketch,
        index_type ld_r_factor) {
        memory::malloc(&dmtx_rp, ld_mtx * num_cols_mtx);
        memory::malloc(&dsketch_rp, ld_sketch * num_cols_sketch);
        memory::malloc(&dresult_rp, ld_r_factor * num_cols_mtx);
        memory::malloc_cpu(&tau, num_rows_sketch);
    }

    void free() {
        memory::free(dmtx_rp);
        memory::free(dsketch_rp);
        memory::free(dresult_rp);
        memory::free_cpu(tau);
    }
};

template <typename value_type_internal, typename value_type,
          typename index_type>
void generate(index_type num_rows_sketch, index_type num_cols_sketch,
              value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx,
              value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
              index_type ld_r_factor, value_type* hat_mtx,
              detail::magma_info& info);

template <typename value_type_internal, typename value_type,
          typename index_type>
void generate(index_type num_rows_sketch, index_type num_cols_sketch,
              value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx,
              value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
              index_type ld_r_factor,
              state<value_type_internal, value_type, index_type>* precond_state,
              detail::magma_info& info, double* runtime, double* t_mm,
              double* t_qr);

}  // namespace gaussian
}  // namespace preconditioner
}  // namespace rls
