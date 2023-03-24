#ifndef SOLVER_HPP 
#define SOLVER_HPP

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

    // AbstractSolver(magma_queue_t queue_in) {
        // queue = queue_in;
    // }

    // ~AbstractSolver();

    // AbstractSolver(double tolerace, magma_int_t iterations, magma_queue_t queue);

    virtual void run() = 0;

    double get_tolerance() { return tolerance_; }

    magma_int_t get_max_iter() { return max_iter_; }

    // magma_queue_t get_queue() {
        // return queue;
    // }
};

class solver : public AbstractSolver {
public:
    // void run() {}

    ~solver() { }

    virtual void generate(std::string& filename_mtx, std::string& filename_rhs);

    virtual void run();

    // {
    //     // if (this->type == Lsqr) {
    //         // switch(combined_value_type)
    //         // {
    //         //     case FP64_FP64:
    //         //     {
    //                 // auto this_solver = dynamic_cast<lsqr<double, double, magma_int_t>*>(this);

    //         //     }
    //         // }
    //     // }
    // }
};

}
}

#endif
