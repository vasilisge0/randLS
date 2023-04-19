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
    std::string input_precond_in_prec = args[2];
    std::string input_precond_prec = args[3];
    std::string input_solver_in_prec = args[4];
    std::string input_solver_prec = args[5];
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
    magma_int_t warmup_iters = std::atoi(input_warmup_iters.c_str());
    magma_int_t runtime_iters = std::atoi(input_runtime_iters.c_str());
    enum GlobalDataType data_type;
    enum GlobalDataType data_type_precond;
    enum GlobalDataType data_type_solver;

    std::shared_ptr<rls::Context<rls::CUDA>> context = rls::Context<rls::CUDA>::create();

    std::shared_ptr<rls::matrix::Dense<double, rls::CUDA>> mtx = rls::matrix::Dense<double, rls::CUDA>::create(context, input_mtx);
    std::shared_ptr<rls::matrix::Dense<double, rls::CUDA>> rhs = rls::matrix::Dense<double, rls::CUDA>::create(context, input_rhs);

    rls::io::print_mtx_gpu(2, 2, mtx->get_values(), 5, mtx->get_context()->get_queue());

    auto con = mtx->get_context();
    auto queue = con->get_queue();

    // Decides the precision of the gaussian preconditioner depending on the inputs.
    auto precond_prec_type = precision_parser(input_precond_prec, input_precond_in_prec);
    std::shared_ptr<rls::preconditioner::generic_preconditioner<rls::CUDA>> precond;
    switch (precond_prec_type) {
        case 0:
        {
            data_type_precond = FP64;
            precond = rls::preconditioner::GeneralizedSplit<double, double, magma_int_t>::create(mtx);
            break;
        }

        case 1:
        {
            data_type_precond = FP64;
            // precond = rls::preconditioner::gaussian<float, double, magma_int_t>::create(mtx);
            break;
        }
        case 2:
        {
            data_type_precond = FP64;
            // precond = rls::preconditioner::gaussian<__half, double, magma_int_t>::create(mtx);
            break;  
        }
        case 3:
        {
            data_type_precond = FP64;
            // precond = rls::preconditioner::gaussian<float, float, magma_int_t>::create(mtx);
            break;
        }
        case 4:
        {
            data_type_precond = FP64;
            // precond = rls::preconditioner::gaussian<__half, float, magma_int_t>::create(mtx);
            break;
        }
        case 5:
        {
            data_type_precond = FP64;
            // precond = rls::preconditioner::gaussian<float, float, magma_int_t>::create(mtx);
            break;
        }
        case 6:
        {
            data_type_precond = FP64;
            // precond = rls::preconditioner::gaussian<float, float, magma_int_t>::create(mtx);
            break;
        }
        default:
            break;
    }

    // Decides the precision of the solver preconditioner depending on the inputs.
    auto solver_prec_type = precision_parser(input_solver_prec, input_solver_in_prec);
    std::shared_ptr<rls::solver::generic_solver<rls::CUDA>> solver;
    switch (solver_prec_type) {
        case 0:
        {
            data_type_solver = FP64;
            solver = rls::solver::Fgmres<double, double, magma_int_t, rls::CUDA>::create(precond.get(), mtx, rhs, tol);
            break;
        }
        case 1:
        {
            data_type_solver = FP64;
            // solver = rls::solver::lsqr<float, double, magma_int_t>::create(precond.get(), mtx, rhs, tol);
            break;
        }
        case 2:
        {
            data_type_solver = FP64;
            // solver = rls::solver::lsqr<__half, double, magma_int_t>::create(precond.get(), mtx, rhs, tol);
            break;  
        }
        case 3:
        {
            data_type_solver = FP64;
            // solver = rls::solver::lsqr<float, double, magma_int_t>::create(precond.get(), mtx, rhs, tol);
            break;
        }
        case 4:
        {
            data_type_solver = FP64;
            // solver = rls::solver::lsqr<float, float, magma_int_t>::create(precond.get(), mtx, rhs, tol);
            break;
        }
        case 5:
        {
            data_type_solver = FP64;
            // solver = rls::solver::lsqr<__half, float, magma_int_t>::create(precond.get(), mtx, rhs, tol);
            break;
        }
        case 6:
        {
            data_type_solver = FP64;
            // solver = rls::solver::lsqr<float, float, magma_int_t>::create(precond.get(), mtx, rhs, tol);
            break;
        }
        default:
            break;
    }

    // Exit if the "outer" precision in both the preconditioner and the solver are different.
    if (data_type_solver != data_type_precond) {
       return 0;
    }

    solver->generate();
    solver->run();
    return 0;
}