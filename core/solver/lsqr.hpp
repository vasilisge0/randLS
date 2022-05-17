#ifndef LSQR_HPP
#define LSQR_HPP


#include "../include/base_types.hpp"


namespace rls {
namespace solver {
namespace lsqr {


template <typename value_type_in, typename value_type, typename index_type>
struct temp_vectors{
    value_type* u;
    value_type* v;
    value_type* w;
    value_type* temp;
    value_type_in* u_in;
    value_type_in* v_in;
    value_type_in* temp_in;
    value_type_in* mtx_in;
    index_type inc;
};

template <typename value_type, typename index_type>
struct temp_scalars{
    value_type alpha;
    value_type beta;
    value_type rho_bar;
    value_type phi_bar;
};

template <typename value_type, typename index_type>
void run(index_type num_rows, index_type num_cols, value_type* mtx,
          value_type* rhs, value_type* init_sol, value_type* sol,
          index_type max_iter, index_type* iter, value_type tol,
          double* resnorm, magma_queue_t queue);

template <typename value_type_in, typename value_type, typename index_type>
void run(index_type num_rows, index_type num_cols, value_type* mtx,
          value_type* rhs, value_type* init_sol, value_type* sol,
          index_type max_iter, index_type* iter, value_type tol,
          double* resnorm, value_type* precond_mtx, index_type ld_precond,
          magma_queue_t queue, double* t_solve);

} // namespace lsqr
} // namespace solver
} // namespace rls


#endif  // run_HPP
