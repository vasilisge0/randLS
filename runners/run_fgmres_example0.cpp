#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>


#include "../include/randls.hpp"


int main(int argc, char* argv[])
{
    // Read input arguments.
    std::vector<std::string> args;
    args.assign(argv, argv + argc);
    std::string input_runfile = args[0];
    std::string input_tol = args[1];
    std::string input_precond_prec = args[2];
    std::string input_precond_in_prec = args[3];
    std::string input_solver_prec = args[4];
    std::string input_solver_in_prec = args[5];
    std::string input_mtx = args[6];
    std::string input_rhs = args[7];
    std::string input_precond_type = args[8];
    std::string input_num_samples = args[9];
    std::string input_outfile = args[10];
    std::string input_warmup_iters = args[11];
    std::string input_runtime_iters = args[12];

//    std::cout << "setting\n";
//    std::cout << "-------\n";
//    std::cout << "        input_runfile: " << input_runfile << '\n';
//    std::cout << "            input_tol: " << input_tol << '\n';
//    std::cout << "input_precond_in_prec: " << input_precond_in_prec << '\n';
//    std::cout << "   input_precond_prec: " << input_precond_prec << '\n';
//    std::cout << " input_solver_in_prec: " << input_solver_in_prec << '\n';
//    std::cout << "    input_solver_prec: " << input_solver_prec << '\n';
//    std::cout << "            input_mtx: " << input_mtx << '\n';
//    std::cout << "   input_precond_type: " << input_precond_type << '\n';
//    std::cout << "    input_num_samples: " << input_num_samples << '\n';
//    std::cout << "        input_outfile: " << input_outfile << '\n';
//    std::cout << "   input_warmup_iters: " << input_warmup_iters << '\n';
//    std::cout << "  input_runtime_iters: " << input_runtime_iters << '\n';
//
//    // Decode input arguments.
//    double tol = std::atof(input_tol.c_str());
//    double sampling_coeff = std::atof(input_num_samples.c_str());
//    magma_int_t warmup_iters = std::atoi(input_warmup_iters.c_str());
//    magma_int_t runtime_iters = std::atoi(input_runtime_iters.c_str());
//
//    // Read input matrices and rhs.
//    using dense = rls::matrix::Dense<rls::CUDA, double>;
//    auto context = rls::share(rls::Context<rls::CUDA>::create());
//    auto mtx = rls::share(dense::create(context, input_mtx));
//    auto rhs = rls::share(dense::create(context, input_rhs));
//    auto sol = rls::share(dense::create(context, rhs->get_size()));
//    sol->zeros();
//
//    // Preconditioner configuration.
//    std::shared_ptr<rls::Preconditioner> precond;
//    rls::preconditioner::logger precond_logger;
//    precond_logger.warmup_runs_ = warmup_iters;
//    precond_logger.runs_ = runtime_iters;
//    using precond_config_type =
//        rls::preconditioner::generalized_split::Config
//            <double, double, double, magma_int_t>;
//    precond_config_type config(sampling_coeff);
//
//    // Sketch configuration.
//    std::string filename_sketch = "sketch.mtx";
//    using sketch_type =
//        rls::GaussianSketch<rls::CUDA, double, double, magma_int_t>;
//    auto sketch = rls::share(sketch_type::create(context, filename_sketch));
//
//    // Constructs preconditioner.
//    using precond_type =
//        rls::preconditioner::GeneralizedSplit
//            <rls::CUDA, double, double, double, magma_int_t>;
//    precond = rls::share(precond_type::create(mtx, sketch, config));
//    precond->compute();
//
//    // Solver configuration.
//    using Fgmres = rls::solver::Fgmres
//        <rls::CUDA, double, double, double, magma_int_t>;
//    using FgmresConfig = rls::solver::iterative::FgmresConfig
//        <double, double, double, magma_int_t>;
//    using Solver = rls::Solver<rls::CUDA>;
//    FgmresConfig solver_config;
//    rls::solver::iterative::Logger solver_logger;
//    solver_config.set_precond(precond);
//    solver_config.set_tolerance(tol);
//    solver_config.set_iterations(3);
//    solver_config.set_logger(solver_logger);
//
//    // Constructs solver and runs.
//    std::shared_ptr<Solver> solver = rls::share(Fgmres::create(context,
//        solver_config, mtx, sol, rhs));
//    solver->run();
//
//    // Writes the following information to an output information file:
//    // 1. matrix dimensinos
//    // 2. maximum number of iterations
//    // 3. sampling coefficient
//    // 4. dimension of sketch
//    // 5. preconditioner runtime
//    // 6. solver runtime
//    // 7. total runtime
//    // 8. completed iterations
//    // 9. residual norm achieved
//    rls::io::write_output(input_outfile.c_str(), mtx->get_size()[0],
//        mtx->get_size()[1], solver_logger.max_iterations_,
//        sampling_coeff, std::ceil(sampling_coeff * mtx->get_size()[1]),
//        precond_logger.runtime_, solver_logger.runtime_,
//        precond_logger.runtime_ + solver_logger.runtime_,
//        solver_logger.completed_iterations_,
//        solver_logger.resnorm_);

    return 0;
}
