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
    Lsqr
};

class AbstractSolver {
protected:
    std::shared_ptr<Context> context_;
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

class generic_solver {
public:
    virtual void generate(std::string& filename_mtx, std::string& filename_rhs) = 0;

    virtual void run() = 0;

    double get_tolerance() { return tolerance_; }

    magma_int_t get_max_iter() { return max_iter_; }

protected:
    std::shared_ptr<Context> context_;
    double tolerance_ = 1e-8;
    magma_int_t max_iter_ = 0;
    bool use_precond_ = false;
    SolverValueType combined_value_type_;
    SolverType type_;
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
