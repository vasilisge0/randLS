#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>


#include "../include/randls.hpp"


int precision_parser(std::string prec, std::string prec_in) {
    if ((prec.compare("fp64") == 0) && (prec_in.compare("fp64") == 0)) {
        return 0;
    }
    else if ((prec.compare("fp64") == 0) && (prec_in.compare("fp32") == 0)) {
        return 1;
    }
    else if ((prec.compare("fp64") == 0) && (prec_in.compare("tf32") == 0)) {
        return 2;
    }
    else if ((prec.compare("fp64") == 0) && (prec_in.compare("fp16") == 0)) {
        return 3;
    }
    else if ((prec.compare("fp32") == 0) && (prec_in.compare("fp32") == 0)) {
        return 4;
    }
    else if ((prec.compare("fp32") == 0) && (prec_in.compare("tf32") == 0)) {
        return 5;
    }
    else if ((prec.compare("fp32") == 0) && (prec_in.compare("fp16") == 0)) {
        return 6;
    }
    return -1;
}

enum GlobalDataType{
    FP64,
    FP32
};

int main(int argc, char* argv[])
{
    //std::vector<std::string> matrix_names = {"sprand_03", "sprand_04", "sparse_medium2", "sparse_large"};
    //std::vector<double> sampling_coeffs = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    //std::vector<double> tolerances = {1e-6, 1e-8, 1e-10};

    //std::vector<std::string> solver_precisions = {"fp64", "fp32"};
    //std::vector<std::string> solver_precisions_internal = {"fp64", "fp32", "bf16", "fp16"};

    //std::vector<std::string> precond_precisions = {"fp64"};
    //std::vector<std::string> precond_precisions_internal = {"fp64", "fp32", "tf32", "fp16"};
    //std::vector<std::string> precond_precisions_apply = {"fp64", "fp32", "bf16", "fp16"};


//    // single instance
//    std::vector<std::string> matrix_names = {"sprand_03"};
//    std::vector<double> sampling_coeffs = {2.5};
//    std::vector<double> tolerances = {1e-8};
//
//    std::vector<std::string> solver_precisions = {"fp64"};
//    std::vector<std::string> solver_precisions_internal = {"fp64"};
//    std::vector<std::string> precond_precisions = {"fp64"};
//    std::vector<std::string> precond_precisions_internal = {"fp64"};
//    std::vector<std::string> precond_precisions_apply = {"fp64"};
//
//    int warmup_iters = 3;
//    int runtime_iters = 3;
//    //using vtype_precond_apply = float;
//    //using vtype_precond_internal = __half;
//   //// using vtype_precond_internal = float;
//    //using vtype_solver_internal = float;
//    //using vtype_solver = float;
//
//    using vtype_precond_apply = double;
//    using vtype_precond_internal = double;
//    using vtype_solver_internal = double;
//    using vtype_solver = double;
//    for (auto & matrix : matrix_names)
//    {
//        for (auto & sampling_coeff : sampling_coeffs)
//        {
//           for (auto & tol : tolerances)
//           {
//               for (auto & precond_precision : precond_precisions)
//               {
//                   for (auto & precond_precision_internal : precond_precisions_internal)
//                   {
//                       for (auto & precond_precision_apply : precond_precisions_apply)
//                       {
//                           for (auto & solver_precision : solver_precisions)
//                           {
//                               for (auto & solver_precision_internal : solver_precisions_internal)
//                               {
//                                   {
//                                        auto & tol = tolerances[0];
//                                        // option 1
//                                        //int maxiter = 15;
//                                        //magma_int_t restarts = 3;
//
//                                        int maxiter = 15;
//                                        magma_int_t restarts = 8;
//                                        //magma_int_t restarts = 1;
//                                        using dense = rls::matrix::Dense<rls::CUDA, double>;
//                                        using sparse = rls::matrix::Sparse<rls::CUDA, double, int>;
//                                        auto context = rls::share(rls::Context<rls::CUDA>::create());
//                                        // Read matrix, rhs and create sol.
//                                        auto input_mtx = "../lsqr_data/" + matrix + ".mtx";
//                                        auto input_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
//                                        auto mtx = rls::share(rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, input_mtx));
//                                        auto rhs = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_rhs));
//                                        auto sol = rls::share(dense::create(context, dim2(mtx->get_size()[1], 1)));
//                                        std::shared_ptr<rls::Preconditioner> precond;
//                                        auto precond_prec_type = precision_parser(precond_precision, precond_precision_internal);
//                                        switch (precond_prec_type) {
//                                            case 0:
//                                            {
//                                                // Construct the sketch matrix.
//                                                dim2 sketch_size = {
//                                                    static_cast<int>(std::ceil(sampling_coeff * mtx->get_size()[1])),
//                                                    mtx->get_size()[0]};
//                                                //auto sketch = rls::share(rls::GaussianSketch<rls::CUDA, double, double, magma_int_t>::create(context, sketch_size));
//                                                using sketch_type =
//                                                    rls::CountSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>;
//                                                size_t k = 1;
//                                                auto sketch = rls::share(sketch_type::create(context, k, sketch_size));
//                                                auto sketch_mtx = sketch->get_mtx();
//                                                {
//                                                    auto tmp = rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, sketch_mtx->get_size(), sketch_mtx->get_nnz());
//                                                    tmp->copy_from(sketch_mtx.get());
//                                                    auto queue = context->get_queue();
//                                                    rls::io::print_mtx_gpu(10, 1, tmp->get_values(), 10, queue);
//                                                }
//
//                                                // Construct preconditioner.
//                                                std::shared_ptr<rls::preconditioner::Config> precond_config;
//                                                using precond_config_type =
//                                                    rls::preconditioner::generalized_split::Config
//                                                        <double, vtype_precond_internal, vtype_precond_apply, magma_int_t>;
//                                                precond_config = rls::share(precond_config_type::create(sampling_coeff));
//                                                precond = rls::preconditioner::GeneralizedSplit<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(mtx, sketch, precond_config);
//                                            }
//                                            default:
//                                            {
//                                                break;
//                                            }
//                                        }
//
//                                        //// ir configuration.
//                                        //std::shared_ptr<rls::Solver<rls::CUDA>> ir;
//
//                                        // Solver configuration.
//                                        auto solver_prec_type = precision_parser(solver_precision, solver_precision_internal);
//                                        using Solver = rls::Solver<rls::CUDA>;
//                                        using SolverLogger = rls::solver::iterative::Logger;
//                                        using SolverConfig = rls::solver::iterative::Config;
//                                        std::shared_ptr<rls::Solver<rls::CUDA>> solver;
//                                        std::shared_ptr<SolverConfig> solver_config;
//                                        SolverLogger solver_logger;
//                                        switch (solver_prec_type) {
//                                            case 0:
//                                            {
//                                                std::cout << "\n\n";
//                                                std::cout << "        matrix: " << matrix << '\n';
//                                                std::cout << "sampling_coeff: " << sampling_coeff << '\n';
//                                                std::cout << "     tolerance: " << tol << '\n';
//                                                using Fgmres = rls::solver::Fgmres
//                                                    <rls::CUDA, vtype_solver, vtype_solver_internal, vtype_precond_apply, magma_int_t>;
//                                                using FgmresConfig = rls::solver::iterative::FgmresConfig
//                                                    <vtype_solver, vtype_solver_internal, vtype_precond_apply, magma_int_t>;
//                                                solver_config = FgmresConfig::create();
//                                                solver_config->set_precond(precond);
//                                                solver_config->set_tolerance(tol);
//                                                //solver_config->set_iterations(maxiter); // @CHECK (!!) need to check with matrix size bounds.
//                                                solver_config->set_logger(solver_logger);
//                                                sol->zeros();
//                                                solver = rls::share(Fgmres::create(context, solver_config, mtx, sol, rhs));
//                                                solver->run();
//                                                //ir = rls::solver::Ir<rls::CUDA, magma_int_t>::create(solver, restarts);
//                                                //ir->run();
//                                            }
//                                            default:
//                                            {
//                                                break;
//                                            }
//                                        }
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
    return 0;
}
