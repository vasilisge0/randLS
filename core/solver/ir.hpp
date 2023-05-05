#ifndef RLS_IR_HPP
#define RLS_IR_HPP


#include <iostream>
#include <memory>
#include <vector>


#include "../include/base_types.hpp"
#include "../preconditioner/gaussian.hpp"
#include "../preconditioner/preconditioner.hpp"


#include "solver.hpp"


namespace rls {
namespace solver {


template <typename index_type, ContextType device_type=CUDA>
class Ir : public generic_solver<device_type> {
public:

    Ir(std::shared_ptr<generic_solver<device_type>> solver, index_type restarts)
    {
        this->solver_ = solver;
        this->context_ = solver->get_context();
        this->restarts_ = restarts;
    }

    // Create method (1) of Fgmres solver
    static std::unique_ptr<Ir<index_type, device_type>>
    create(std::shared_ptr<generic_solver<device_type>> solver, index_type restarts)
    {
        return std::unique_ptr<Ir<index_type, device_type>>(
            new Ir<index_type, device_type>(
                solver, restarts));
    }

    // Allocates matrices used in fgmres and constructs preconditioner.
    void generate()
    {
        solver_->generate();
    }

    void run()
    {
        for (auto r = 0; r < restarts_; r++) {
            std::cout << "restart: " << r << '\n';
            solver_->run();
        }
    }

    void run_with_logger()
    {

    }

private:
    std::shared_ptr<generic_solver<device_type>> solver_;
    index_type restarts_;
};


}  // namespace solver
}  // namespace rls


#endif  // RLS_FGMRES_HPP
