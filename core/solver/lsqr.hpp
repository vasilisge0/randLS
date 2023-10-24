#ifndef RLS_LSQR_HPP
#define RLS_LSQR_HPP


#include "../include/base_types.hpp"
#include "solver.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "../preconditioner/sketchqr.hpp"
#include "../matrix/mtxop.hpp"

#include <iostream>
#include <memory>


namespace rls {
namespace solver {



namespace iterative {

template<typename vtype,
         typename vtype_internal,
         typename vtype_precond_apply,
         typename vtype_refine,
         typename itype>
class LsqrConfig : public Config {

    LsqrConfig<vtype, vtype_internal, vtype_precond_apply, vtype_refine, itype>& with_restarts(int restarts) {
        restarts_ = restarts;
    }

private:
    int restarts_ = 0;
};

} // end of namespace iterative


namespace lsqr {


template <ContextType device, typename vtype, typename vtype_internal,
          typename vtype_precond_apply, typename vtype_refine, typename itype>
struct Workspace {
    // Fields used for storing vectors and matrices used by run_lsqr.
    //vtype* u       = nullptr;
    //vtype* v       = nullptr;
    //vtype* w       = nullptr;
    //vtype* temp    = nullptr;
    //vtype_internal* u_in    = nullptr;
    //vtype_internal* v_in    = nullptr;
    //vtype_internal* temp_in = nullptr;
    //vtype_internal* mtx_in  = nullptr;
    std::shared_ptr<rls::matrix::Dense<device, vtype>> u;
    std::shared_ptr<rls::matrix::Dense<device, vtype>> v;
    std::shared_ptr<rls::matrix::Dense<device, vtype>> w;
    std::shared_ptr<rls::matrix::Dense<device, vtype>> temp;
    std::shared_ptr<rls::matrix::Dense<device, vtype_internal>> u_in;
    std::shared_ptr<rls::matrix::Dense<device, vtype_internal>> v_in;
    std::shared_ptr<rls::matrix::Dense<device, vtype_internal>> temp_in;
    std::shared_ptr<rls::matrix::Dense<device, vtype_internal>> mtx_in;
    std::shared_ptr<rls::MtxOp<device>> mtx_apply;
    std::shared_ptr<rls::MtxOp<device>> mtx_;
    std::shared_ptr<rls::matrix::Dense<device, vtype>> res_;
    std::shared_ptr<rls::matrix::Dense<device, vtype>> sol_;
    std::shared_ptr<rls::matrix::Dense<device, vtype_refine>> rhs_refine_;
    std::shared_ptr<rls::matrix::Dense<device, vtype_refine>> temp_refine_;
    std::shared_ptr<rls::matrix::Dense<device, vtype>> temp1;
    std::shared_ptr<rls::matrix::Dense<device, vtype_precond_apply>> u_apply_;
    std::shared_ptr<rls::matrix::Dense<device, vtype_precond_apply>> v_apply_;
    std::shared_ptr<rls::matrix::Dense<device, vtype_refine>> true_sol_;
    std::shared_ptr<rls::matrix::Dense<device, vtype_refine>> noisy_sol_;
    std::shared_ptr<rls::matrix::Dense<device, vtype_refine>> true_error_;

    // Scalar metadata variables.
    itype inc         = 1;
    itype num_elems_0 = 0;
    itype num_elems_1 = 0;
    // Floating point scalars used in run_lsqr.
    vtype alpha   = 0.0;
    vtype beta    = 0.0;
    vtype rho_bar = 0.0;
    vtype phi_bar = 0.0;
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

    vtype_precond_apply* t = nullptr;

    ~Workspace();

    Workspace(std::shared_ptr<Context<device>> context, dim2 size);

    Workspace(std::shared_ptr<Context<device>> context, std::shared_ptr<MtxOp<device>> mtx,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs);

    Workspace(std::shared_ptr<Context<device>> context, std::shared_ptr<MtxOp<device>> mtx,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> true_sol,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> true_error);

    Workspace(std::shared_ptr<Context<device>> context, std::shared_ptr<MtxOp<device>> mtx,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> true_sol,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> true_error,
              std::shared_ptr<matrix::Dense<device, vtype_refine>> noisy_sol);

    static std::shared_ptr<Workspace<device, vtype, vtype_internal, vtype_precond_apply, vtype_refine, itype>>
        create(std::shared_ptr<Context<device>> context, dim2 size);

    static std::shared_ptr<Workspace<device, vtype, vtype_internal, vtype_precond_apply, vtype_refine, itype>>
        create(std::shared_ptr<Context<device>> context, std::shared_ptr<MtxOp<device>> mtx,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs);

    static std::shared_ptr<Workspace<device, vtype, vtype_internal, vtype_precond_apply, vtype_refine, itype>>
        create(std::shared_ptr<Context<device>> context, std::shared_ptr<MtxOp<device>> mtx,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> true_sol,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> true_error);

    static std::shared_ptr<Workspace<device, vtype, vtype_internal, vtype_precond_apply, vtype_refine, itype>>
        create(std::shared_ptr<Context<device>> context, std::shared_ptr<MtxOp<device>> mtx,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> true_sol,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> true_error,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> noisy_sol);

    double compute_stagnation_index()
    {
        //std::cout << std::abs(resnorm_ - resnorm_previous_) / stagnation_weight << '\n';
        //return (std::abs(resnorm_ - resnorm_previous_) < (stagnation_weight * stagnation_tolerance) && (resnorm_ < resnorm_previous_));
        double stagnation_index = std::abs(resnorm - resnorm_previous);
        return stagnation_index;
    }

    bool solver_stagnates(double stagnation_index, double stagnation_tolerance, double stagnation_weight)
    {
        //std::cout << std::abs(resnorm_ - resnorm_previous_) / stagnation_weight << '\n';
        //return (std::abs(resnorm_ - resnorm_previous_) < (stagnation_weight * stagnation_tolerance) && (resnorm_ < resnorm_previous_));
        return (stagnation_index < (stagnation_weight * stagnation_tolerance));
    }

};


}


template<ContextType device,
         typename vtype,
         typename vtype_internal_0,
         typename vtype_precond_apply,
         typename vtype_refine,
         typename itype>
class Lsqr : virtual public Solver<device> {
public:

    static std::unique_ptr<Lsqr<device, vtype, vtype_internal_0,
        vtype_precond_apply, vtype_refine, itype>>
        create(std::shared_ptr<Context<device>> context,
               std::shared_ptr<iterative::Config> config,
               std::shared_ptr<MtxOp<device>> mtx,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
               std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs)
    {
        auto t = std::static_pointer_cast<iterative::LsqrConfig<vtype, vtype_internal_0, vtype_precond_apply, vtype_refine, itype>>(config);
        return std::unique_ptr<Lsqr<device, vtype, vtype_internal_0, vtype_precond_apply, vtype_refine, itype>>(
            new Lsqr<device, vtype, vtype_internal_0, vtype_precond_apply, vtype_refine, itype>(context, t, mtx, sol, rhs));
    }

    void run();

    bool stagnates();

    bool converges();

    iterative::Logger get_logger();

private:

    Lsqr(std::shared_ptr<Context<device>> context,
         std::shared_ptr<iterative::LsqrConfig<vtype, vtype_internal_0, vtype_precond_apply, vtype_refine, itype>> config,
         std::shared_ptr<MtxOp<device>> mtx,
         std::shared_ptr<matrix::Dense<device, vtype_refine>> sol,
         std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs);

    void generate();

    std::shared_ptr<iterative::Logger> logger_; // this changed
    std::shared_ptr<Context<device>> context_;
    std::shared_ptr<iterative::LsqrConfig<vtype, vtype_internal_0, vtype_precond_apply, vtype_refine, itype>> config_;
    std::shared_ptr<lsqr::Workspace<device, vtype, vtype_internal_0, vtype_precond_apply, vtype_refine, itype>> workspace_;
    std::shared_ptr<MtxOp<device>> mtx_;
    std::shared_ptr<matrix::Dense<device, vtype_refine>> rhs_;
    std::shared_ptr<matrix::Dense<device, vtype_refine>> sol_;
    std::shared_ptr<matrix::Dense<device, vtype_refine>> true_sol_;
    std::shared_ptr<matrix::Dense<device, vtype_refine>> true_error_;
};


}   // end of solver namespace
}   // end of lsqr namespace

#endif  // RLS_LSQR_HPP
