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


int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    args.assign(argv, argv + argc);
    std::string input_runfile  = args[0];
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

    std::cout << "setting\n";
    std::cout << "-------\n";
    std::cout << "        input_runfile: " << input_runfile << '\n';
    std::cout << "            input_tol: " << input_tol << '\n';
    std::cout << "input_precond_in_prec: " << input_precond_in_prec << '\n';
    std::cout << "   input_precond_prec: " << input_precond_prec << '\n';
    std::cout << " input_solver_in_prec: " << input_solver_in_prec << '\n';
    std::cout << "    input_solver_prec: " << input_solver_prec << '\n';
    std::cout << "            input_mtx: " << input_mtx << '\n';
    std::cout << "   input_precond_type: " << input_precond_type << '\n';
    std::cout << "    input_num_samples: " << input_num_samples << '\n';
    std::cout << "        input_outfile: " << input_outfile << '\n';
    std::cout << "   input_warmup_iters: " << input_warmup_iters << '\n';
    std::cout << "  input_runtime_iters: " << input_runtime_iters << '\n';






    // Decode input arguments.
    double tol = std::atof(input_tol.c_str());
    double sampling_coeff = std::atof(input_num_samples.c_str());

    // Sets input data
    auto context = rls::Context<rls::CUDA>::create();

    // read matrices
    auto mtx = rls::matrix::Dense<float>::create(context, input_mtx);
    auto rhs = rls::matrix::Dense<double>::create(context, input_rhs);

    auto precond = rls::preconditioner::GeneralizedSplit<double, double, magma_int_t>::create(mtx, sampling_coeff);

    std::shared_ptr<rls::solver::generic_solver<rls::CUDA>> solver;
    solver = rls::solver::Fgmres<float, double, magma_int_t, rls::CUDA>::create(precond.get(), mtx, rhs, tol);

    // Generate preconditioner and solver
    solver->generate();
    solver->run();


    return 0;
}
