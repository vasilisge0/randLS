#ifndef RLS_SOLVER_HPP 
#define RLS_SOLVER_HPP

#include "../dense/dense.hpp"
#include "magma_v2.h"
#include <iostream>


namespace rls {
namespace solver {


enum SolverValueType {
    Undefined_PrecondValueType,
    FP64_FP64,
    FP32_FP64,
    TF32_FP64,
    FP16_FP64,
    FP32_FP32,
    TF32_FP32,
    FP16_FP32
};

enum SolverType {
    Undefined_SolverType,
    LSQR,
    FGMRES
};

template <ContextType device_type=CUDA>
class AbstractSolver {
protected:
    std::shared_ptr<Context<device_type>> context_;
    double tolerance_ = 1e-8;
    magma_int_t max_iter_ = 0;
    bool use_precond_ = false;
    SolverValueType combined_value_type_;
    SolverType type_;

public:
    AbstractSolver() {}

    virtual void run() = 0;

    double get_tolerance() { return tolerance_; }

    magma_int_t get_max_iter() { return max_iter_; }
};

// class solver : public AbstractSolver {
// public:
    // ~solver() { }
// 
    // virtual void generate(std::string& filename_mtx, std::string& filename_rhs);
// 
    // virtual void run();
// };

template <ContextType device_type=CUDA>
class generic_solver {
public:
    virtual void generate() = 0;

    virtual void run() = 0;

    double get_tolerance() { return tolerance_; }

    double get_runtime() { return runtime_; }

    double get_resnorm() { return resnorm_; }

    magma_int_t get_iterations_completed() { return iter_; }

    magma_int_t get_max_iter() { return max_iter_; }

    std::shared_ptr<Context<device_type>> get_context() { return context_; }

protected:
    std::shared_ptr<Context<device_type>> context_;
    double tolerance_ = 1e-8;
    magma_int_t max_iter_ = 0;
    bool use_precond_ = false;
    SolverValueType combined_value_type_;
    SolverType type_;
    magma_int_t iter_ = 0;
    double resnorm_ = 1.0;
    double runtime_ = 0.0;
};

// class solver : public AbstractSolver {
// public:
// 
    // virtual void generate(std::string& filename_mtx, std::string& filename_rhs);
// 
    // virtual void run();
// };

}   // end of namespace solver
}   // end of namespace rls

#endif
