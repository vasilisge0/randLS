#ifndef RLS_FGMRES_HPP
#define RLS_FGMRES_HPP


#include <iostream>
#include <memory>
#include <vector>


#include "../../utils/io.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "../preconditioner/sketchqr.hpp"
#include "../include/base_types.hpp"
#include "solver.hpp"


namespace rls {
namespace solver {
namespace iterative {


template <typename value_type_0, typename value_type_in_0,
          typename value_type_apply_0, typename index_type>
class FgmresConfig : public Config {
    FgmresConfig<value_type_0, value_type_in_0, value_type_apply_0, index_type>&
    with_restarts(int restarts)
    {
        restarts_ = restarts;
    }

private:
    int restarts_ = 0;
};

template class FgmresConfig<double, double, double, magma_int_t>;
template class FgmresConfig<double, float, double, magma_int_t>;


}  // end of namespace iterative


namespace fgmres {


// Vectors used by Fgmres method.
template <ContextType device_type, typename value_type,
          typename value_type_internal_0, typename value_type_precond_0,
          typename index_type>
struct WorkspaceSparse
{
    std::shared_ptr<Context<device_type>> context_;
    // Scalars.
    index_type inc = 1;
    index_type max_iter_ = 0;
    // Vectors and matrices.
    //using dense = gko::matrix::Dense<value_type>;
    using dense = matrix::Dense<device_type, value_type>;
    using dense_cpu = matrix::Dense<CPU, value_type>;
    std::shared_ptr<dense> u;                     // is this needed?
    std::shared_ptr<dense> v_basis;               // this is the v_basis stored in transpose.
    std::shared_ptr<dense> w;                     // is this needed?
    std::shared_ptr<dense> temp;                  // used at fgmres::spmv.
    std::shared_ptr<dense> residual;              // this is the residual of the augmented system.
    //using dense_internal_0 = gko::matrix::Dense<value_type_internal_0>;
    using dense_internal_0 = matrix::Dense<device_type, value_type_internal_0>;
    std::shared_ptr<dense_internal_0> w_in;       // internal precision w.
    std::shared_ptr<dense_internal_0> v_in;       // internal precision v.
    std::shared_ptr<dense_internal_0> z_in;       // internal precision z.
    std::shared_ptr<dense_internal_0> temp_in;    // internal precision temp vector.
    //std::shared_ptr<dense_internal_0> mtx_in;   // internal precision matrix(?) -> this should be changed to a sparse one.
    std::shared_ptr<MtxOp<device_type>> mtx_in;   // internal precision matrix(?) -> this should be changed to a sparse one.
    std::shared_ptr<MtxOp<device_type>> mtx_in_t; // internal precision matrix(?) -> this should be changed to a sparse one.
    std::shared_ptr<dense_cpu> hessenberg_mtx;    // coefficient matrix
    std::shared_ptr<dense_cpu> hessenberg_rhs;    // reduced rhs
    std::shared_ptr<dense> hessenberg_mtx_gpu;    // coefficient matrix gpu
    std::shared_ptr<dense> hessenberg_rhs_gpu;    // reduced rhs gpu
    std::shared_ptr<dense> z_basis;               // transpose z_basis
    std::shared_ptr<dense> sol0 = nullptr;
    std::shared_ptr<std::vector<std::pair<value_type, value_type>>>
        givens_cache;
    std::shared_ptr<dense_cpu> tmp_cpu; // cpu memory
    value_type alpha;
    value_type beta;
    value_type rho_bar;
    value_type phi_bar;
    value_type h;
    std::shared_ptr<dense> aug_sol;
    std::shared_ptr<dense> aug_residual;
    std::shared_ptr<dense> aug_rhs;
    // Used for storing residual norms.
    double rhsnorm = 0.0;
    double resnorm = 0.0;
    double resnorm_previous = 0.0;
    // iterations
    magma_int_t completed_iterations = 0;
    magma_int_t completed_iterations_per_restart = 0;
    magma_int_t completed_restarts = 0;
    //
    bool new_restart_active = false;

    static std::shared_ptr<
        WorkspaceSparse<device_type, value_type, value_type_internal_0,
                  value_type_precond_0, index_type>>
    create(std::shared_ptr<Context<device_type>> context,
           std::shared_ptr<MtxOp<device_type>> mtx,
           std::shared_ptr<iterative::FgmresConfig<value_type, value_type_internal_0,
                                   value_type_precond_0, index_type>>
               config, dim2 size);

    WorkspaceSparse(std::shared_ptr<Context<device_type>> context,
              std::shared_ptr<MtxOp<device_type>> mtx,
              std::shared_ptr<iterative::FgmresConfig<value_type, value_type_internal_0,
                                      value_type_precond_0, index_type>>
                  config, dim2 size);
};

//// Specialized gemv operation for Fgmres on the generalized system
//// [I A; A' 0] for input m x n matrix A and m >> n.
//template <typename device_type, typename value_type, typename index_type>
//void gemv(std::shared_ptr<device_type> context, magma_trans_t trans, index_type num_rows, index_type num_cols,
//          value_type alpha, value_type* mtx, index_type ld,
//          value_type* u_vector, index_type inc_u, value_type beta,
//          value_type* v_vector, index_type inc_v, value_type* tmp);

//// Specialized gemv operation for Fgmres on the generalized system
//// [I A; A' 0] for input m x n matrix A and m >> n.
//template <typename device_type, typename value_type, typename index_type>
//void gemv(std::shared_ptr<device_type> context, magma_trans_t trans, index_type num_rows, index_type num_cols,
//          gko::matrix::Dense<value_type>* alpha, gko::LinOp* mtx,
//          gko::matrix::Dense<value_type>* u_vector_part_0,
//          gko::matrix::Dense<value_type>* u_vector_part_1,
//          gko::matrix::Dense<value_type>* beta,
//          gko::matrix::Dense<value_type>* v_vector_part_0,
//          gko::matrix::Dense<value_type>* v_vector_part_1,
//          gko::matrix::Dense<value_type>* tmp);

template <typename device_type, typename value_type, typename index_type>
void gemv(std::shared_ptr<device_type> context, //1
          magma_trans_t trans,//2
          index_type num_rows,//3
          index_type num_cols,//4
          gko::matrix::Dense<value_type>* alpha,//5
          gko::LinOp* mtx,//6
          gko::matrix::Dense<value_type>* u_vector,//7
          //gko::matrix::Dense<value_type>* u_vector_part_0,
          //gko::matrix::Dense<value_type>* u_vector_part_1,
          gko::matrix::Dense<value_type>* beta,//8
          gko::matrix::Dense<value_type>* v_vector,//9
          //gko::matrix::Dense<value_type>* v_vector_part_0,
          //gko::matrix::Dense<value_type>* v_vector_part_1,
          gko::matrix::Dense<value_type>* tmp);//10


}  // end of namespace fgmres


template <ContextType device_type, typename value_type,
          typename value_type_internal_0, typename value_type_precond_0,
          typename index_type>
class Fgmres : public Solver<device_type>
{

public:

    static std::unique_ptr<
        Fgmres<device_type, value_type, value_type_internal_0,
               value_type_precond_0, index_type>>
    create(std::shared_ptr<Context<device_type>> context,
           std::shared_ptr<iterative::Config> config,
           std::shared_ptr<MtxOp<device_type>> mtx,
           std::shared_ptr<matrix::Dense<device_type, value_type>> sol,
           std::shared_ptr<matrix::Dense<device_type, value_type>> rhs);

    void run();

    bool stagnates();

    bool converges();

    iterative::Logger get_logger();

private:

    Fgmres(std::shared_ptr<Context<device_type>> context,
           std::shared_ptr<iterative::FgmresConfig<value_type, value_type_internal_0,
                                   value_type_precond_0, index_type>> config,
           std::shared_ptr<MtxOp<device_type>> mtx,
           std::shared_ptr<matrix::Dense<device_type, value_type>> sol,
           std::shared_ptr<matrix::Dense<device_type, value_type>> rhs);

    void run_with_logger();

    std::shared_ptr<iterative::FgmresConfig<value_type, value_type_internal_0,
                            value_type_precond_0, index_type>> config_;
    std::shared_ptr<
        fgmres::WorkspaceSparse<device_type, value_type, value_type_internal_0,
                          value_type_precond_0, index_type>>
        workspace_;
    std::shared_ptr<MtxOp<device_type>> mtx_;
    std::shared_ptr<matrix::Dense<device_type, value_type>> dmtx_;
    std::shared_ptr<matrix::Dense<device_type, value_type>> rhs_;
    std::shared_ptr<matrix::Dense<device_type, value_type>> sol_;
    std::shared_ptr<matrix::Dense<device_type, value_type>> glb_rhs_;
    std::shared_ptr<matrix::Dense<device_type, value_type>> glb_sol_;
    std::shared_ptr<iterative::Logger> logger_;
};


}  // end of namespace solver
}  // end of namespace rls


#endif  // RLS_FGMRES_HPP
