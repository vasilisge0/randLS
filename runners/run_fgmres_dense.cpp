#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>


#include <ginkgo/ginkgo.hpp>
#include "../include/randls.hpp"
#include "runner_helper.hpp"


int main(int argc, char* argv[])
{
//    // Read input arguments.
//    std::vector<std::string> args;
//    args.assign(argv, argv + argc);
//    std::string input_runfile = args[0];
//    std::string input_tol = args[1];
//    std::string input_precond_prec = args[2];
//    std::string input_precond_in_prec = args[3];
//    std::string input_solver_prec = args[4];
//    std::string input_solver_in_prec = args[5];
//    std::string input_mtx = args[6];
//    std::string input_rhs = args[7];
//    std::string input_precond_type = args[8];
//    std::string input_num_samples = args[9];
//    std::string input_outfile = args[10];
//    std::string input_warmup_iters = args[11];
//    std::string input_runtime_iters = args[12];
//
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
//    using sparse = rls::matrix::Sparse<rls::CUDA, double, int>;
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
//    std::shared_ptr<rls::preconditioner::Config> precond_config;
//    auto precond_prec_type =
//        precision_parser(input_precond_prec, input_precond_in_prec);
//    switch (precond_prec_type) {
//        case 0:
//        {
//            using precond_config_type =
//                rls::preconditioner::generalized_split::Config
//                    <double, double, double, magma_int_t>;
//            precond_config = rls::share(precond_config_type::create(sampling_coeff));
//        }
//        case 1:
//        {
//            using precond_config_type =
//                rls::preconditioner::generalized_split::Config
//                    <double, float, double, magma_int_t>;
//            precond_config = rls::share(precond_config_type::create(sampling_coeff));
//        }
//        case 2:
//        {
//            using precond_config_type =
//                rls::preconditioner::generalized_split::Config
//                    <double, float, double, magma_int_t>;
//            precond_config = rls::share(precond_config_type::create(sampling_coeff));
//        }
//        case 3:
//        {
//            using precond_config_type =
//                rls::preconditioner::generalized_split::Config
//                    <double, __half, double, magma_int_t>;
//            precond_config = rls::share(precond_config_type::create(sampling_coeff));
//        }
//        case 4:
//        {
//            using precond_config_type =
//                rls::preconditioner::generalized_split::Config
//                    <float, float, float, magma_int_t>;
//            precond_config = rls::share(precond_config_type::create(sampling_coeff));
//        }
//        case 5:
//        {
//            using precond_config_type =
//                rls::preconditioner::generalized_split::Config
//                    <float, float, float, magma_int_t>;
//            precond_config = rls::share(precond_config_type::create(sampling_coeff));
//        }
//        case 6:
//        {
//            using precond_config_type =
//                rls::preconditioner::generalized_split::Config
//                    <float, __half, float, magma_int_t>;
//            precond_config = rls::share(precond_config_type::create(sampling_coeff));
//        }
//        default:
//        {
//            break;
//        }
//    }
//
//    // Sketch configuration.
//    sampling_coeff = 3.0;
//    dim2 sketch_size = {
//        static_cast<int>(std::ceil(sampling_coeff * mtx->get_size()[1])),
//        mtx->get_size()[0]};
//    using sketch_type =
//        rls::GaussianSketch<rls::CUDA, double, double, magma_int_t>;
//    auto sketch = rls::share(sketch_type::create(context, sketch_size));
//
//    // Constructs preconditioner.
//    using precond_type =
//        rls::preconditioner::GeneralizedSplit
//            <rls::CUDA, double, double, double, magma_int_t>;
//    precond = rls::share(precond_type::create(mtx, sketch, precond_config));
//    precond->generate();
//    //auto precond_mtx = static_cast<rls::preconditioner::GeneralizedSplit<rls::CUDA, double, double, double, magma_int_t>*>(precond.get())->get_precond();
//    //rls::io::write_mtx("sparse_precond.mtx", precond_mtx->get_size()[0],
//    //    precond_mtx->get_size()[1], precond_mtx->get_values());
//
//    // Solver configuration.
//    using Solver = rls::Solver<rls::CUDA>;
//    using SolverLogger = rls::solver::iterative::Logger;
//    using SolverConfig = rls::solver::iterative::Config;
//    std::shared_ptr<Solver> solver;
//    std::shared_ptr<SolverConfig> solver_config;
//    SolverLogger solver_logger;
////
////    auto solver_prec_type =
////        precision_parser(input_solver_prec, input_solver_in_prec);
////    switch(solver_prec_type) {
////        case 0:
////        {
//            using Fgmres = rls::solver::Fgmres
//                <rls::CUDA, double, double, double, magma_int_t>;
//            using FgmresConfig = rls::solver::iterative::FgmresConfig
//                <double, double, double, magma_int_t>;
//            //precond_mtx->eye();
//            solver_config = FgmresConfig::create();
//            solver_config->set_precond(precond);
//            solver_config->set_tolerance(tol);
//            solver_config->set_iterations(1000);         // @CHECK (!!) need to check with matrix size bounds.
//            solver_config->set_logger(solver_logger);
//            solver = rls::share(Fgmres::create(context, solver_config, mtx, sol, rhs));
////            break;
////        }
////        case 1:
////        {
////            using Fgmres = rls::solver::Fgmres
////                <rls::CUDA, double, double, double, magma_int_t>;
////            using FgmresConfig = rls::solver::iterative::FgmresConfig
////                <double, double, double, magma_int_t>;
////            solver_config = FgmresConfig::create();
////            solver = rls::share(Fgmres::create(context, solver_config, mtx, sol, rhs));
////            break;
////        }
////        case 2:
////        {
////            using Fgmres = rls::solver::Fgmres
////                <rls::CUDA, double, float, double, magma_int_t>;
////            using FgmresConfig = rls::solver::iterative::FgmresConfig
////                <double, float, double, magma_int_t>;
////            solver_config = FgmresConfig::create();
////            solver = rls::share(Fgmres::create(context, solver_config, mtx, sol, rhs));
////            break;
////        }
////        case 3:
////        {
////            using Fgmres = rls::solver::Fgmres
////                <rls::CUDA, double, __half, double, magma_int_t>;
////            using FgmresConfig = rls::solver::iterative::FgmresConfig
////                <double, __half, double, magma_int_t>;
////            solver_config = FgmresConfig::create();
////            solver = rls::share(Fgmres::create(context, solver_config, mtx, sol, rhs));
////            break;
////        }
////        case 4:
////        {
////            using Fgmres = rls::solver::Fgmres
////                <rls::CUDA, float, float, float, magma_int_t>;
////            using FgmresConfig = rls::solver::iterative::FgmresConfig
////                <float, float, float, magma_int_t>;
////            solver_config = FgmresConfig::create();
////
////            auto mtx_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(mtx->get_context(), mtx->get_size()));
////            mtx_in->copy_from(mtx);
////
////            auto sol_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(sol->get_context(), sol->get_size()));
////            sol_in->copy_from(sol);
////
////            auto rhs_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(rhs->get_context(), rhs->get_size()));
////            rhs_in->copy_from(rhs);
////
////            solver = rls::share(Fgmres::create(context, solver_config, mtx_in, sol_in, rhs_in));
////            break;
////        }
////        case 5:
////        {
////            using Fgmres = rls::solver::Fgmres
////                <rls::CUDA, float, float, float, magma_int_t>;
////            using FgmresConfig = rls::solver::iterative::FgmresConfig
////                <float, float, float, magma_int_t>;
////            solver_config = FgmresConfig::create();
////
////            auto mtx_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(mtx->get_context(), mtx->get_size()));
////            mtx_in->copy_from(mtx);
////
////            auto sol_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(sol->get_context(), sol->get_size()));
////            sol_in->copy_from(sol);
////
////            auto rhs_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(rhs->get_context(), rhs->get_size()));
////            rhs_in->copy_from(rhs);
////
////            solver = rls::share(Fgmres::create(context, solver_config, mtx_in, sol_in, rhs_in));
////            break;
////        }
////        case 6:
////        {
////            using Fgmres = rls::solver::Fgmres
////                <rls::CUDA, float, __half, float, magma_int_t>;
////            using FgmresConfig = rls::solver::iterative::FgmresConfig
////                <float, __half, float, magma_int_t>;
////            solver_config = FgmresConfig::create();
////
////            auto mtx_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(mtx->get_context(), mtx->get_size()));
////            mtx_in->copy_from(mtx);
////
////            auto sol_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(sol->get_context(), sol->get_size()));
////            sol_in->copy_from(sol);
////
////            auto rhs_in = rls::share(rls::matrix::Dense<rls::CUDA, float>::create(rhs->get_context(), rhs->get_size()));
////            rhs_in->copy_from(rhs);
////
////            solver = rls::share(Fgmres::create(context, solver_config, mtx_in, sol_in, rhs_in));
////            break;
////        }
////        default:
////        {
////            break;
////        }
////    }
////
//
//    solver->run();  // Constructs solver and runs.
//
////    //// Writes the following information to an output information file:
////    //// 1. matrix dimensions
////    //// 2. maximum number of iterations
////    //// 3. sampling coefficient
////    //// 4. dimension of sketch
////    //// 5. preconditioner runtime
////    //// 6. solver runtime
////    //// 7. total runtime
////    //// 8. completed iterations
////    //// 9. residual norm achieved
////    //rls::io::write_output(input_outfile.c_str(), mtx->get_size()[0],
////    //    mtx->get_size()[1], solver_logger.max_iterations_,
////    //    sampling_coeff, std::ceil(sampling_coeff * mtx->get_size()[1]),
////    //    precond_logger.runtime_, solver_logger.runtime_,
////    //    precond_logger.runtime_ + solver_logger.runtime_,
////    //    solver_logger.completed_iterations_,
////    //    solver_logger.resnorm_);
//
    return 0;
}
