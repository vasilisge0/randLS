#include <cstdlib>
#include <iostream>
#include <vector>


#include "../include/randls.hpp"


// Maps precision arguments (strings) to numerical values.
int precision_parser(std::string prec, std::string prec_in)
{
    if ((prec.compare("fp64") == 0) && (prec_in.compare("fp64") == 0)) {
        return 0;
    } else if ((prec.compare("fp64") == 0) && (prec_in.compare("fp32") == 0)) {
        return 1;
    } else if ((prec.compare("fp64") == 0) && (prec_in.compare("tf32") == 0)) {
        return 2;
    } else if ((prec.compare("fp64") == 0) && (prec_in.compare("fp16") == 0)) {
        return 3;
    } else if ((prec.compare("fp32") == 0) && (prec_in.compare("fp32") == 0)) {
        return 4;
    } else if ((prec.compare("fp32") == 0) && (prec_in.compare("tf32") == 0)) {
        return 5;
    } else if ((prec.compare("fp32") == 0) && (prec_in.compare("fp16") == 0)) {
        return 6;
    }
    return -1;
}

// Stores data used for experiments.
struct lsqr {
    rls::detail::magma_info magma_config;
    magma_int_t num_rows = 0;
    magma_int_t num_cols = 0;
    magma_int_t sampled_rows = 0;
    magma_int_t max_iter = num_rows;
    magma_int_t iter = 0;
    magma_int_t argc = 0;
    magma_int_t warmup_iters = 0;
    magma_int_t runtime_iters = 0;
    void* mtx = nullptr;
    void* dmtx = nullptr;
    void* sol = nullptr;
    void* init_sol = nullptr;
    void* rhs = nullptr;
    void* precond_mtx = nullptr;
    double sampling_coeff = 1.01;
    double t_precond = 0.0;
    double t_solve = 0.0;
    double t_total = 0.0;
    double t_mm = 0.0;
    double t_qr = 0.0;
    double t_precond_avg = 0.0;
    double t_solve_avg = 0.0;
    double t_total_avg = 0.0;
    double t_mm_avg = 0.0;
    double t_qr_avg = 0.0;
    double tol = 1e-6;
    double relres_norm = 0.0;
    double relres_norm_avg = 0.0;
    bool use_precond = false;
    std::string filename_out;
    std::vector<std::string> args;

    void run();

    void dispatch_preconditioner();

    void dispatch_solver();

    void print_runtime_info();

    void write_output();

    void initialize();

    void finalize();
};

// Selects the version of the preconditioner to be used.
void lsqr::dispatch_preconditioner()
{
    auto first_index = 1;
    std::string filename_mtx = args[first_index + 4];
    std::string filename_rhs = args[first_index + 5];
    std::cout << "  filename_mtx: " << filename_mtx << '\n';
    std::cout << "  filename_rhs: " << filename_rhs << '\n';
    std::cout << "sampling_coeff: " << sampling_coeff << '\n';

    if (args[first_index + 6].compare("precond") == 0) {
        use_precond = true;
        sampling_coeff = std::atof(args[first_index + 7].c_str());
    }

    switch (precision_parser(args[first_index], args[first_index + 1])) {
    case 0:
        rls::utils::initialize_with_precond<double, double, int>(
            filename_mtx, filename_rhs, &num_rows, &num_cols, (double**)&mtx,
            (double**)&dmtx, (double**)&init_sol, (double**)&sol,
            (double**)&rhs, sampling_coeff, &sampled_rows,
            (double**)&precond_mtx, magma_config, &t_precond, &t_mm, &t_qr);
        break;

    case 1:
        rls::utils::initialize_with_precond<float, double, int>(
            filename_mtx, filename_rhs, &num_rows, &num_cols, (double**)&mtx,
            (double**)&dmtx, (double**)&init_sol, (double**)&sol,
            (double**)&rhs, sampling_coeff, &sampled_rows,
            (double**)&precond_mtx, magma_config, &t_precond, &t_mm, &t_qr);
        break;

    case 2:
        rls::detail::use_tf32_math_operations(magma_config);
        rls::utils::initialize_with_precond<float, double, int>(
            filename_mtx, filename_rhs, &num_rows, &num_cols, (double**)&mtx,
            (double**)&dmtx, (double**)&init_sol, (double**)&sol, (double**)&rhs,
            sampling_coeff, &sampled_rows, (double**)&precond_mtx, magma_config,
            &t_precond, &t_mm, &t_qr);
        rls::detail::disable_tf32_math_operations(magma_config);
        break;

    case 3:
        rls::utils::initialize_with_precond<__half, double, int>(
            filename_mtx, filename_rhs, &num_rows, &num_cols, (double**)&mtx,
            (double**)&dmtx, (double**)&init_sol, (double**)&sol,
            (double**)&rhs, sampling_coeff, &sampled_rows,
            (double**)&precond_mtx, magma_config, &t_precond, &t_mm, &t_qr);
        break;

    case 4:
        rls::utils::initialize_with_precond<float, float, int>(
            filename_mtx, filename_rhs, &num_rows, &num_cols, (float**)&mtx,
            (float**)&dmtx, (float**)&init_sol, (float**)&sol, (float**)&rhs,
            sampling_coeff, &sampled_rows, (float**)&precond_mtx, magma_config,
            &t_precond, &t_mm, &t_qr);
        break;

    case 5:
        rls::detail::use_tf32_math_operations(magma_config);
        rls::utils::initialize_with_precond<float, float, int>(
            filename_mtx, filename_rhs, &num_rows, &num_cols, (float**)&mtx,
            (float**)&dmtx, (float**)&init_sol, (float**)&sol, (float**)&rhs,
            sampling_coeff, &sampled_rows, (float**)&precond_mtx, magma_config,
            &t_precond, &t_mm, &t_qr);
        rls::detail::disable_tf32_math_operations(magma_config);
        break;

    case 6:
        rls::utils::initialize_with_precond<__half, float, int>(
            filename_mtx, filename_rhs, &num_rows, &num_cols, (float**)&mtx,
            (float**)&dmtx, (float**)&init_sol, (float**)&sol, (float**)&rhs,
            sampling_coeff, &sampled_rows, (float**)&precond_mtx, magma_config,
            &t_precond, &t_mm, &t_qr);
        break;

    default:
        std::cout << "Exitting without running lsqr." << '\n';
        break;
    }
}

// Selects the version of the solver to be used.
void lsqr::dispatch_solver()
{
    auto first_index = 3;
    max_iter = num_rows;
    iter = 0;
    tol = std::atof(args[0].c_str());
    relres_norm = 0.0;
    switch (precision_parser(args[first_index], args[first_index + 1])) {
    case 0:
        rls::solver::lsqr::run<double>(
            num_rows, num_cols, (double*)dmtx, (double*)rhs, (double*)init_sol,
            (double*)sol, max_iter, &iter, tol, &relres_norm,
            (double*)precond_mtx, sampled_rows, magma_config.queue, &t_solve);
        rls::utils::finalize_with_precond(
            (double*)mtx, (double*)dmtx, (double*)init_sol, (double*)sol,
            (double*)rhs, (double*)precond_mtx, magma_config);
        break;
    case 1:
        rls::solver::lsqr::run<float>(
            num_rows, num_cols, (double*)dmtx, (double*)rhs, (double*)init_sol,
            (double*)sol, max_iter, &iter, tol, &relres_norm,
            (double*)precond_mtx, sampled_rows, magma_config.queue, &t_solve);
        rls::utils::finalize_with_precond(
            (double*)mtx, (double*)dmtx, (double*)init_sol, (double*)sol,
            (double*)rhs, (double*)precond_mtx, magma_config);
        break;

    case 2:
        rls::detail::use_tf32_math_operations(magma_config);
        rls::solver::lsqr::run<float>(
            num_rows, num_cols, (double*)dmtx, (double*)rhs, (double*)init_sol,
            (double*)sol, max_iter, &iter, tol, &relres_norm,
            (double*)precond_mtx, sampled_rows, magma_config.queue, &t_solve);
        rls::detail::disable_tf32_math_operations(magma_config);
        rls::utils::finalize_with_precond(
            (double*)mtx, (double*)dmtx, (double*)init_sol, (double*)sol,
            (double*)rhs, (double*)precond_mtx, magma_config);
        break;

    case 3:
        rls::solver::lsqr::run<__half>(
            num_rows, num_cols, (double*)dmtx, (double*)rhs, (double*)init_sol,
            (double*)sol, max_iter, &iter, tol, &relres_norm,
            (double*)precond_mtx, sampled_rows, magma_config.queue, &t_solve);
        rls::utils::finalize_with_precond(
            (double*)mtx, (double*)dmtx, (double*)init_sol, (double*)sol,
            (double*)rhs, (double*)precond_mtx, magma_config);
        break;

    case 4:
        rls::solver::lsqr::run<float, float, magma_int_t>(
            num_rows, num_cols, (float*)dmtx, (float*)rhs, (float*)init_sol,
            (float*)sol, max_iter, &iter, tol, &relres_norm,
            (float*)precond_mtx, sampled_rows, magma_config.queue, &t_solve);
        rls::utils::finalize_with_precond(
            (float*)mtx, (float*)dmtx, (float*)init_sol, (float*)sol,
            (float*)rhs, (float*)precond_mtx, magma_config);
        break;

    case 5:
        rls::detail::use_tf32_math_operations(magma_config);
        rls::solver::lsqr::run<float, float, magma_int_t>(
            num_rows, num_cols, (float*)dmtx, (float*)rhs, (float*)init_sol,
            (float*)sol, max_iter, &iter, tol, &relres_norm,
            (float*)precond_mtx, sampled_rows, magma_config.queue, &t_solve);
        rls::detail::disable_tf32_math_operations(magma_config);
        rls::utils::finalize_with_precond(
            (float*)mtx, (float*)dmtx, (float*)init_sol, (float*)sol,
            (float*)rhs, (float*)precond_mtx, magma_config);
        break;

    case 6:
        rls::solver::lsqr::run<__half, float, magma_int_t>(
            num_rows, num_cols, (float*)dmtx, (float*)rhs, (float*)init_sol,
            (float*)sol, max_iter, &iter, tol, &relres_norm,
            (float*)precond_mtx, sampled_rows, magma_config.queue, &t_solve);
        rls::utils::finalize_with_precond(
            (float*)mtx, (float*)dmtx, (float*)init_sol, (float*)sol,
            (float*)rhs, (float*)precond_mtx, magma_config);
        break;

    default:
        std::cout << "No option specified for solver.\n";
        break;
    }
}

void lsqr::print_runtime_info()
{
    std::cout << "inputs:" << '\n';
    std::cout << "=======\n";
    std::cout << "         precond precision: " << args[1] << '\n';
    std::cout << "internal precond precision: " << args[2] << '\n';
    std::cout << "          solver precision: " << args[3] << '\n';
    std::cout << " solver internal precision: " << args[4] << '\n';
    std::cout << "                    matrix: " << args[5] << '\n';
    std::cout << "                       rhs: " << args[6] << '\n';
    std::cout << "      sampling coefficient: " << sampling_coeff << '\n'
              << '\n';

    std::cout << "runtimes:\n";
    std::cout << "=========\n";
    std::cout << "         warmup iterations: " << warmup_iters << '\n';
    std::cout << "        runtime iterations: " << runtime_iters << '\n';
    std::cout << "          precond time_avg: " << t_precond_avg << '\n';
    std::cout << "            solve time_avg: " << t_solve_avg << '\n';
    std::cout << "            total time_avg: " << t_total_avg << '\n';
    std::cout << "                  t_mm_avg: " << t_mm_avg << '\n';
    std::cout << "                  t_qr_avg: " << t_qr_avg << '\n';
    std::cout << "                      iter: " << iter << '\n';
    std::cout << "                relres_avg: " << relres_norm_avg << '\n';
    std::cout << "      sampling coefficient: " << sampling_coeff << '\n';
    std::cout << "              sampled rows: " << sampled_rows << '\n';
    std::cout << "               output file: " << filename_out << '\n';
}

void lsqr::write_output()
{
    rls::io::write_output((char*)filename_out.c_str(), num_rows, num_cols,
                                max_iter, sampling_coeff, sampled_rows,
                                t_precond_avg, t_solve_avg, t_total_avg,
                                t_mm_avg, t_qr_avg, iter, relres_norm_avg);
}

void lsqr::run()
{
    filename_out = args[9];
    warmup_iters = std::atoi(args[10].c_str());
    runtime_iters = std::atoi(args[11].c_str());

    // Warmup runs.
    for (auto i = 0; i < warmup_iters; i++) {
        t_precond = 0.0;
        t_solve = 0.0;
        t_mm = 0.0;
        t_qr = 0.0;
        dispatch_preconditioner();
        dispatch_solver();
        std::cout << "  warmup -> t_precond: " << t_precond << '\n';
        std::cout << "  warmup ->   t_solve: " << t_solve << '\n';
    }

    // Runs for measuring runtime.
    t_precond_avg = 0.0;
    t_solve_avg = 0.0;
    t_mm_avg = 0.0;
    t_qr_avg = 0.0;
    for (auto i = 0; i < runtime_iters; i++) {
        t_precond = 0.0;
        t_solve = 0.0;
        t_mm = 0.0;
        t_qr = 0.0;
        dispatch_preconditioner();
        dispatch_solver();
        t_precond_avg += t_precond;
        t_solve_avg += t_solve;
        t_mm_avg += t_mm;
        t_qr_avg += t_qr;
        std::cout << "    t_precond: " << t_precond << '\n';
        std::cout << "      t_solve: " << t_solve << '\n';
        std::cout << "t_precond_avg: " << t_precond_avg << "[t_mm: " << t_mm
                  << ", t_qr:" << t_qr << "]" << '\n';
        std::cout << "  t_solve_avg: " << t_solve_avg << '\n';
        relres_norm_avg += relres_norm;
    }
    relres_norm_avg /= runtime_iters;           // relative residual
    t_precond_avg /= runtime_iters;             // precond runtime
    t_solve_avg /= runtime_iters;               // solve runtime
    t_total_avg = t_precond_avg + t_solve_avg;  // total runtime
    t_mm_avg /= runtime_iters;  // matrix-mult runtime (part of precond)
    t_qr_avg /= runtime_iters;  // qr runtime (part of precond)
}

void lsqr::initialize() { rls::detail::configure_magma(magma_config); }

void lsqr::finalize()
{
    cudaStreamDestroy(magma_config.cuda_stream);
    cublasDestroy(magma_config.cublas_handle);
    cusparseDestroy(magma_config.cusparse_handle);
    curandDestroyGenerator(magma_config.rand_generator);
    magma_queue_destroy(magma_config.queue);
    magma_finalize();
}

int main(int argc, char* argv[])
{
    lsqr solver;
    solver.args.assign(argv + 1, argv + argc);
    solver.initialize();
    solver.run();
    solver.print_runtime_info();
    // solver.write_output();
    solver.finalize();
    return 0;
}
