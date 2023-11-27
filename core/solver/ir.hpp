#ifndef RLS_IR_HPP
#define RLS_IR_HPP


#include <iostream>
#include <memory>
#include <vector>


#include "../preconditioner/sketchqr.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "../include/base_types.hpp"
#include "solver.hpp"


namespace rls {
namespace solver {


template <ContextType device,
          typename itype>
class Ir : public Solver<device> {
public:

    static std::unique_ptr<Ir<device, itype>>
    create(std::shared_ptr<Solver<device>> solver, itype restarts)
    {
        return std::unique_ptr<Ir<device, itype>>(
            new Ir<device, itype>(
                solver, restarts));
    }

    void generate()
    {
        solver_->generate();
    }

    void run()
    {
        for (auto r = 0; r < restarts_; r++) {
            solver_->run();
            std::cout << "r: " << r << "/" << restarts_ << '\n';
            if (solver_->converges()) {
                break;
            }
        }
    }

    bool stagnates()
    {
        return solver_->stagnates();
    }

    bool converges()
    {
        return solver_->converges();
    }


    iterative::Logger get_logger()
    {
        return solver_->get_logger();
    }

private:

    Ir(std::shared_ptr<Solver<device>> solver, itype restarts) : Solver<device>(solver->get_context())
    {
        this->solver_ = solver;
        this->restarts_ = restarts;
    }

    std::shared_ptr<Solver<device>> solver_;
    itype restarts_;
};


}  // namespace solver
}  // namespace rls


#endif  // RLS_FGMRES_HPP
