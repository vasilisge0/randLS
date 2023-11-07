#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <iomanip>

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

template<typename vtype_precond_apply, typename vtype_precond_internal, typename vtype_solver_internal, typename vtype_solver>
void run_sparse_lsqr_instance(double tol, int maxiter, int miniter, magma_int_t restarts, double sampling_coeff, std::string& matrix, std::string& filename_relres,
    std::string& filename_true_error, std::string& filename_stagnation, std::string& filename_matrix, std::string& filename_rhs, std::string& filename_true_sol)
{
    using dense = rls::matrix::Dense<rls::CUDA, double>;
    using sparse = rls::matrix::Sparse<rls::CUDA, double, int>;
    auto context = rls::share(rls::Context<rls::CUDA>::create());
    // Read matrix, rhs and create sol.
    auto input_mtx = filename_matrix;
    auto input_rhs = filename_rhs;
    auto mtx = rls::share(rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, input_mtx));
    auto rhs = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_rhs));
    auto sol = rls::share(dense::create(context, dim2(mtx->get_size()[1], 1)));
    sol->zeros();
    std::shared_ptr<rls::Preconditioner> precond;

    // -------- Preconditioner construction -------- //

    // @error here in half.
    {
        // Construct the sketch matrix.
        dim2 sketch_size = {
            static_cast<int>(std::ceil(sampling_coeff * mtx->get_size()[1])),
            mtx->get_size()[0]};
        //auto sketch = rls::share(rls::GaussianSketch<rls::CUDA, double, double, magma_int_t>::create(context, sketch_size));
        using sketch_type =
            rls::CountSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>;
        size_t k = 1;
        auto sketch = rls::share(sketch_type::create(context, k, sketch_size));
        //auto sketch_mtx = sketch->get_mtx();
        //{
        //    auto tmp = rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, sketch_mtx->get_size(), sketch_mtx->get_nnz());
        //    tmp->copy_from(sketch_mtx.get());
        //}

        // Construct preconditioner.
        std::shared_ptr<rls::preconditioner::Config> precond_config;
        using precond_config_type =
            rls::preconditioner::SketchQrConfig
                <double, vtype_precond_internal, vtype_precond_apply, magma_int_t>;
        precond_config = rls::share(precond_config_type::create(sampling_coeff));
        //@error in next line in half precision
        precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(mtx, sketch, precond_config);
        //std::string str = "../lsqr_data/sprand_03_precond.mtx";
        //precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(
        //    mtx, str, precond_config);

        //auto p = static_cast<rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>*>(precond.get());
        ////auto P = static_cast<rls::matrix::Dense<rls::CUDA, double>*>(p->get_precond_mtx().get());
        //auto P = p->get_precond_mtx();
        //auto queue = context->get_queue();
        //rls::io::write_mtx("../lsqr_data/sprand_03_precond.mtx", P->get_size()[0], P->get_size()[1],
        //    P->get_values(), P->get_size()[0], queue);
    }

    // -------- solver construction -------- //

    std::shared_ptr<rls::Solver<rls::CUDA>> ir;
    using Solver = rls::Solver<rls::CUDA>;
    using SolverLogger = rls::solver::iterative::Logger;
    using SolverConfig = rls::solver::iterative::Config;
    std::shared_ptr<rls::Solver<rls::CUDA>> solver;
    std::shared_ptr<SolverConfig> solver_config;

    // Configure and run solver.
    {
        std::cout << "\n\n";
        std::cout << "        matrix: " << matrix << '\n';
        std::cout << "sampling_coeff: " << sampling_coeff << '\n';
        std::cout << "     tolerance: " << tol << '\n';
        //auto input_true_sol = "../lsqr_data/" + matrix + "_xtrue.mtx";
        auto input_true_sol = filename_true_sol;
        using Lsqr = rls::solver::Lsqr
            <rls::CUDA, vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
        using LsqrConfig = rls::solver::iterative::LsqrConfig
            <vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
        solver_config = LsqrConfig::create();
        solver_config->set_precond(precond);
        solver_config->set_tolerance(tol);
        solver_config->set_stagnation_tolerance(1e-3);  // <= sqrt(gemv precision)
        solver_config->set_restarts(restarts);
        solver_config->set_min_iterations(miniter);
        solver_config->set_iterations(maxiter);
        solver_config->set_filename_relres(filename_relres);
        solver_config->set_filename_true_error(input_true_sol, filename_true_error);
        solver_config->set_filename_stagnation(filename_stagnation);
std::cout << "here\n";
        sol->zeros();
std::cout << "before solver\n";
        solver = rls::share(Lsqr::create(context, solver_config, mtx, sol, rhs));
std::cout << "after solver\n";
        ir = rls::solver::Ir<rls::CUDA, magma_int_t>::create(solver, restarts);
        ir->run();
        //auto logger = ir->get_logger();
        //logger.write_history();
    }
}

template<typename vtype_precond_apply, typename vtype_precond_internal, typename vtype_solver_internal, typename vtype_solver>
void run_dense_lsqr_instance(double tol, int maxiter, int miniter, magma_int_t restarts, double sampling_coeff, std::string& matrix, std::string& filename_relres,
    std::string& filename_true_error, std::string& filename_stagnation, std::string& filename_matrix, std::string& filename_rhs, std::string& filename_true_sol)
{
    using dense = rls::matrix::Dense<rls::CUDA, double>;
    using sparse = rls::matrix::Sparse<rls::CUDA, double, int>;
    auto context = rls::share(rls::Context<rls::CUDA>::create());
    // Read matrix, rhs and create sol.
    auto input_mtx = filename_matrix;
    auto input_rhs = filename_rhs;
    auto mtx = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_mtx));
    auto rhs = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_rhs));
    auto sol = rls::share(dense::create(context, dim2(mtx->get_size()[1], 1)));
    sol->zeros();
    std::shared_ptr<rls::Preconditioner> precond;

    // -------- Preconditioner construction -------- //

    // @error here in half.
    {
        // Construct the sketch matrix.
        dim2 sketch_size = {
            static_cast<int>(std::ceil(sampling_coeff * mtx->get_size()[1])),
            mtx->get_size()[0]};
        auto sketch = rls::share(rls::GaussianSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>::create(context, sketch_size));
        //using sketch_type =
        //    rls::CountSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>;
        //size_t k = 1;
        //auto sketch = rls::share(sketch_type::create(context, k, sketch_size));
        //auto sketch_mtx = sketch->get_mtx();
        //{
        //    auto tmp = rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, sketch_mtx->get_size(), sketch_mtx->get_nnz());
        //    tmp->copy_from(sketch_mtx.get());
        //}

        // Construct preconditioner.
        std::shared_ptr<rls::preconditioner::Config> precond_config;
        using precond_config_type =
            rls::preconditioner::SketchQrConfig
                <double, vtype_precond_internal, vtype_precond_apply, magma_int_t>;
        precond_config = rls::share(precond_config_type::create(sampling_coeff));
        //@error in next line in half precision
        precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(mtx, sketch, precond_config);
        //std::string str = "../lsqr_data/sprand_03_precond.mtx";
        //precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(
        //    mtx, str, precond_config);

        //auto p = static_cast<rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>*>(precond.get());
        ////auto P = static_cast<rls::matrix::Dense<rls::CUDA, double>*>(p->get_precond_mtx().get());
        //auto P = p->get_precond_mtx();
        //auto queue = context->get_queue();
        //rls::io::write_mtx("../lsqr_data/sprand_03_precond.mtx", P->get_size()[0], P->get_size()[1],
        //    P->get_values(), P->get_size()[0], queue);
    }

    // -------- solver construction -------- //

    std::shared_ptr<rls::Solver<rls::CUDA>> ir;
    using Solver = rls::Solver<rls::CUDA>;
    using SolverLogger = rls::solver::iterative::Logger;
    using SolverConfig = rls::solver::iterative::Config;
    std::shared_ptr<rls::Solver<rls::CUDA>> solver;
    std::shared_ptr<SolverConfig> solver_config;

    // Configure and run solver.
    {
        std::cout << "\n\n";
        std::cout << "        matrix: " << matrix << '\n';
        std::cout << "sampling_coeff: " << sampling_coeff << '\n';
        std::cout << "     tolerance: " << tol << '\n';
        //auto input_true_sol = "../lsqr_data/" + matrix + "_xtrue.mtx";
        auto input_true_sol = filename_true_sol;
        using Lsqr = rls::solver::Lsqr
            <rls::CUDA, vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
        using LsqrConfig = rls::solver::iterative::LsqrConfig
            <vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
        solver_config = LsqrConfig::create();
        solver_config->set_precond(precond);
        solver_config->set_tolerance(tol);
        solver_config->set_stagnation_tolerance(1e-3);  // <= sqrt(gemv precision)
        solver_config->set_restarts(restarts);
        solver_config->set_min_iterations(miniter);
        solver_config->set_iterations(maxiter);
        solver_config->set_filename_relres(filename_relres);
        solver_config->set_filename_true_error(input_true_sol, filename_true_error);
        solver_config->set_filename_stagnation(filename_stagnation);
std::cout << "here\n";
        sol->zeros();
std::cout << "before solver\n";
        solver = rls::share(Lsqr::create(context, solver_config, mtx, sol, rhs));
std::cout << "after solver\n";
        ir = rls::solver::Ir<rls::CUDA, magma_int_t>::create(solver, restarts);
        ir->run();
        auto logger = ir->get_logger();
        logger.write_history();
    }
}

template<typename vtype_precond_apply, typename vtype_precond_internal, typename vtype_solver_internal, typename vtype_solver>
void run_dense_reg_lsqr_instance(double tol, int maxiter, int miniter, magma_int_t restarts, double sampling_coeff, std::string& matrix, std::string& filename_relres,
    std::string& filename_true_error, std::string& filename_stagnation, std::string& filename_noisy_error, std::string& filename_true_similarity,
    std::string& filename_noisy_similarity, std::string& filename_matrix, std::string& filename_rhs, std::string& filename_true_sol, std::string& filename_noisy_sol)
{
    using dense = rls::matrix::Dense<rls::CUDA, double>;
    using sparse = rls::matrix::Sparse<rls::CUDA, double, int>;
    auto context = rls::share(rls::Context<rls::CUDA>::create());
    // Read matrix, rhs and create sol.
    //auto input_mtx = "../lsqr_data/" + matrix + ".mtx";
    //auto input_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    std::cout << "filename_rhs: " << filename_rhs << '\n';
    auto input_mtx = filename_matrix;
    auto input_rhs = filename_rhs;
    auto mtx = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_mtx));
    std::cout << "mtx->get_size()[0]: " << mtx->get_size()[0] << '\n';
    auto rhs = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_rhs));
    auto sol = rls::share(dense::create(context, dim2(mtx->get_size()[1], 1)));
    sol->zeros();
    std::cout << "sol->get_size()[0]: " << sol->get_size()[0] << '\n';
    std::cout << "sol->get_ld(): " << sol->get_ld() << '\n';
    std::shared_ptr<rls::Preconditioner> precond;

    // -------- Preconditioner construction -------- //

    // @error here in half.
    {
        // Construct the sketch matrix.
        dim2 sketch_size = {
            static_cast<int>(std::ceil(sampling_coeff * mtx->get_size()[1])),
            mtx->get_size()[0]};
        auto sketch = rls::share(rls::GaussianSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>::create(context, sketch_size));
        //using sketch_type =
        //    rls::CountSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>;
        //size_t k = 1;
        //auto sketch = rls::share(sketch_type::create(context, k, sketch_size));
        //auto sketch_mtx = sketch->get_mtx();
        //{
        //    auto tmp = rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, sketch_mtx->get_size(), sketch_mtx->get_nnz());
        //    tmp->copy_from(sketch_mtx.get());
        //}

        // Construct preconditioner.
        std::shared_ptr<rls::preconditioner::Config> precond_config;
        using precond_config_type =
            rls::preconditioner::SketchQrConfig
                <double, vtype_precond_internal, vtype_precond_apply, magma_int_t>;
        precond_config = rls::share(precond_config_type::create(sampling_coeff));
        std::cout << "before CREATE\n";
        //@error in next line in half precision
        precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(mtx, sketch, precond_config);
        //std::string str = "../lsqr_data/sprand_03_precond.mtx";
        //precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(
        //    mtx, str, precond_config);

        //auto p = static_cast<rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>*>(precond.get());
        ////auto P = static_cast<rls::matrix::Dense<rls::CUDA, double>*>(p->get_precond_mtx().get());
        //auto P = p->get_precond_mtx();
        //auto queue = context->get_queue();
        //rls::io::write_mtx("../lsqr_data/sprand_03_precond.mtx", P->get_size()[0], P->get_size()[1],
        //    P->get_values(), P->get_size()[0], queue);
    }

    // -------- solver construction -------- //

    std::shared_ptr<rls::Solver<rls::CUDA>> ir;
    using Solver = rls::Solver<rls::CUDA>;
    using SolverLogger = rls::solver::iterative::Logger;
    using SolverConfig = rls::solver::iterative::Config;
    std::shared_ptr<rls::Solver<rls::CUDA>> solver;
    std::shared_ptr<SolverConfig> solver_config;

    // Configure and run solver.
    {
        std::cout << "\n\n";
        std::cout << "        matrix: " << matrix << '\n';
        std::cout << "sampling_coeff: " << sampling_coeff << '\n';
        std::cout << "     tolerance: " << tol << '\n';
        //auto input_true_sol = "../lsqr_data/" + matrix + "_xtrue.mtx";
        auto input_true_sol = filename_true_sol;
        auto input_noisy_sol = filename_noisy_sol;
        using Lsqr = rls::solver::Lsqr
            <rls::CUDA, vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
        using LsqrConfig = rls::solver::iterative::LsqrConfig
            <vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
        solver_config = LsqrConfig::create();
        solver_config->set_precond(precond);
        solver_config->set_tolerance(tol);
        solver_config->set_stagnation_tolerance(1e-3);  // <= sqrt(gemv precision)
        solver_config->set_restarts(restarts);
        solver_config->set_min_iterations(miniter);
        solver_config->set_iterations(maxiter);
        solver_config->set_filename_relres(filename_relres);
        solver_config->set_filename_true_error(input_true_sol, filename_true_error);
        solver_config->set_filename_noisy_error(input_noisy_sol, filename_noisy_error);
        solver_config->set_filename_stagnation(filename_stagnation);
        solver_config->set_filename_similarity(filename_true_similarity, filename_noisy_similarity);
        sol->zeros();
        solver = rls::share(Lsqr::create(context, solver_config, mtx, sol, rhs));
        ir = rls::solver::Ir<rls::CUDA, magma_int_t>::create(solver, restarts);
        ir->run();
        auto logger = ir->get_logger();
        logger.write_history();
    }
}

template<typename vtype_precond_apply, typename vtype_precond_internal, typename vtype_solver_internal, typename vtype_solver>
void run_sparse_reg_lsqr_instance(double tol, int maxiter, int miniter, magma_int_t restarts, double sampling_coeff, std::string& matrix, std::string& filename_relres,
    std::string& filename_true_error, std::string& filename_stagnation, std::string& filename_noisy_error, std::string& filename_true_similarity,
    std::string& filename_noisy_similarity, std::string& filename_matrix, std::string& filename_rhs, std::string& filename_true_sol, std::string& filename_noisy_sol)
{
    using dense = rls::matrix::Dense<rls::CUDA, double>;
    using sparse = rls::matrix::Sparse<rls::CUDA, double, int>;
    auto context = rls::share(rls::Context<rls::CUDA>::create());
    // Read matrix, rhs and create sol.
    //auto input_mtx = "../lsqr_data/" + matrix + ".mtx";
    //auto input_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    std::cout << "filename_rhs: " << filename_rhs << '\n';
    auto input_mtx = filename_matrix;
    auto input_rhs = filename_rhs;
    auto mtx = rls::share(rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, input_mtx));
    std::cout << "mtx->get_size()[0]: " << mtx->get_size()[0] << '\n';
    auto rhs = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_rhs));
    auto sol = rls::share(dense::create(context, dim2(mtx->get_size()[1], 1)));
    sol->zeros();
    std::cout << "sol->get_size()[0]: " << sol->get_size()[0] << '\n';
    std::cout << "sol->get_ld(): " << sol->get_ld() << '\n';
    std::shared_ptr<rls::Preconditioner> precond;

    // -------- Preconditioner construction -------- //

    // @error here in half.
    {
        // Construct the sketch matrix.
        dim2 sketch_size = {
            static_cast<int>(std::ceil(sampling_coeff * mtx->get_size()[1])),
            mtx->get_size()[0]};
        auto sketch = rls::share(rls::GaussianSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>::create(context, sketch_size));
        //using sketch_type =
        //    rls::CountSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>;
        //size_t k = 1;
        //auto sketch = rls::share(sketch_type::create(context, k, sketch_size));
        //auto sketch_mtx = sketch->get_mtx();
        //{
        //    auto tmp = rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, sketch_mtx->get_size(), sketch_mtx->get_nnz());
        //    tmp->copy_from(sketch_mtx.get());
        //}

        // Construct preconditioner.
        std::shared_ptr<rls::preconditioner::Config> precond_config;
        using precond_config_type =
            rls::preconditioner::SketchQrConfig
                <double, vtype_precond_internal, vtype_precond_apply, magma_int_t>;
        precond_config = rls::share(precond_config_type::create(sampling_coeff));
        std::cout << "before CREATE\n";
        //@error in next line in half precision
        precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(mtx, sketch, precond_config);
        std::cout << "AFTER CREATE\n";
        //std::string str = "../lsqr_data/sprand_03_precond.mtx";
        //precond = rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(
        //    mtx, str, precond_config);

        //auto p = static_cast<rls::preconditioner::SketchQr<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>*>(precond.get());
        ////auto P = static_cast<rls::matrix::Dense<rls::CUDA, double>*>(p->get_precond_mtx().get());
        //auto P = p->get_precond_mtx();
        //auto queue = context->get_queue();
        //rls::io::write_mtx("../lsqr_data/sprand_03_precond.mtx", P->get_size()[0], P->get_size()[1],
        //    P->get_values(), P->get_size()[0], queue);
    }

    //// -------- solver construction -------- //

    //std::shared_ptr<rls::Solver<rls::CUDA>> ir;
    //using Solver = rls::Solver<rls::CUDA>;
    //using SolverLogger = rls::solver::iterative::Logger;
    //using SolverConfig = rls::solver::iterative::Config;
    //std::shared_ptr<rls::Solver<rls::CUDA>> solver;
    //std::shared_ptr<SolverConfig> solver_config;

    //// Configure and run solver.
    //{
    //    std::cout << "\n\n";
    //    std::cout << "        matrix: " << matrix << '\n';
    //    std::cout << "sampling_coeff: " << sampling_coeff << '\n';
    //    std::cout << "     tolerance: " << tol << '\n';
    //    //auto input_true_sol = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //    auto input_true_sol = filename_true_sol;
    //    auto input_noisy_sol = filename_noisy_sol;
    //    using Lsqr = rls::solver::Lsqr
    //        <rls::CUDA, vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
    //    using LsqrConfig = rls::solver::iterative::LsqrConfig
    //        <vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
    //    solver_config = LsqrConfig::create();
    //    solver_config->set_precond(precond);
    //    solver_config->set_tolerance(tol);
    //    solver_config->set_stagnation_tolerance(1e-3);  // <= sqrt(gemv precision)
    //    solver_config->set_restarts(restarts);
    //    solver_config->set_min_iterations(miniter);
    //    solver_config->set_iterations(maxiter);
    //    solver_config->set_filename_relres(filename_relres);
    //    solver_config->set_filename_true_error(input_true_sol, filename_true_error);
    //    solver_config->set_filename_noisy_error(input_noisy_sol, filename_noisy_error);
    //    solver_config->set_filename_stagnation(filename_stagnation);
    //    solver_config->set_filename_similarity(filename_true_similarity, filename_noisy_similarity);
    //    sol->zeros();
    //    solver = rls::share(Lsqr::create(context, solver_config, mtx, sol, rhs));
    //    ir = rls::solver::Ir<rls::CUDA, magma_int_t>::create(solver, restarts);
    //    ir->run();
    //    auto logger = ir->get_logger();
    //    logger.write_history();
    //}
}

template<typename vtype_precond_apply, typename vtype_precond_internal, typename vtype_solver_internal, typename vtype_solver>
void run_fgmres_instance(double tol, int maxiter, int miniter, magma_int_t restarts, double sampling_coeff, std::string& matrix, std::string& filename_relres,
    std::string& filename_true_error, std::string& filename_stagnation)
{
    using dense = rls::matrix::Dense<rls::CUDA, double>;
    using sparse = rls::matrix::Sparse<rls::CUDA, double, int>;
    auto context = rls::share(rls::Context<rls::CUDA>::create());
    // Read matrix, rhs and create sol.
    auto input_mtx = "../lsqr_data/" + matrix + ".mtx";
    auto input_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    auto mtx = rls::share(rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, input_mtx));
    auto rhs = rls::share(rls::matrix::Dense<rls::CUDA, double>::create(context, input_rhs));
    auto sol = rls::share(dense::create(context, dim2(mtx->get_size()[1], 1)));
    sol->zeros();
    std::cout << "sol->get_size()[0]: " << sol->get_size()[0] << '\n';
    std::cout << "sol->get_ld(): " << sol->get_ld() << '\n';
    std::shared_ptr<rls::Preconditioner> precond;

    // -------- Preconditioner construction -------- //
    std::cout << " in run fgmres\n";

    // @error here in half.
    {
        // Construct the sketch matrix.
        dim2 sketch_size = {
            static_cast<int>(std::ceil(sampling_coeff * mtx->get_size()[1])),
            mtx->get_size()[0]};
        //auto sketch = rls::share(rls::GaussianSketch<rls::CUDA, double, double, magma_int_t>::create(context, sketch_size));
        using sketch_type =
            rls::CountSketch<rls::CUDA, double, vtype_precond_internal, magma_int_t>;
        size_t k = 1;
        auto sketch = rls::share(sketch_type::create(context, k, sketch_size));
        auto sketch_mtx = sketch->get_mtx();
        {
            auto tmp = rls::matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, sketch_mtx->get_size(), sketch_mtx->get_nnz());
            tmp->copy_from(sketch_mtx.get());
        }
        // Construct preconditioner.
        std::shared_ptr<rls::preconditioner::Config> precond_config;
        using precond_config_type =
            rls::preconditioner::generalized_split::Config
                <double, vtype_precond_internal, vtype_precond_apply, magma_int_t>;
        precond_config = rls::share(precond_config_type::create(sampling_coeff));
        precond = rls::preconditioner::GeneralizedSplit<rls::CUDA, double, vtype_precond_internal, vtype_precond_apply, magma_int_t>::create(mtx, sketch, precond_config);
    }

    {
        // Solver configuration.
        using Solver = rls::Solver<rls::CUDA>;
        using SolverLogger = rls::solver::iterative::Logger;
        using SolverConfig = rls::solver::iterative::Config;
        std::shared_ptr<rls::Solver<rls::CUDA>> solver;
        std::shared_ptr<rls::Solver<rls::CUDA>> ir;
        std::shared_ptr<SolverConfig> solver_config;
        std::cout << "\n\n";
        std::cout << "        matrix: " << matrix << '\n';
        std::cout << "sampling_coeff: " << sampling_coeff << '\n';
        std::cout << "     tolerance: " << tol << '\n';
        using Fgmres = rls::solver::Fgmres
            <rls::CUDA, vtype_solver, vtype_solver_internal, vtype_precond_apply, magma_int_t>;
        using FgmresConfig = rls::solver::iterative::FgmresConfig
            <vtype_solver, vtype_solver_internal, vtype_precond_apply, magma_int_t>;
        solver_config = FgmresConfig::create();
        solver_config->set_precond(precond);
        solver_config->set_tolerance(tol);
        solver_config->set_precond(precond);
        solver_config->set_tolerance(tol);
        solver_config->set_stagnation_tolerance(1e-3);  // <= sqrt(gemv precision)
        solver_config->set_restarts(restarts);
        solver_config->set_min_iterations(miniter);
        solver_config->set_iterations(maxiter);
        sol->zeros();
        solver = rls::share(Fgmres::create(context, solver_config, mtx, sol, rhs));
        //solver->run();
        ir = rls::solver::Ir<rls::CUDA, magma_int_t>::create(solver, restarts);
        ir->run();
    }

    //// -------- solver construction -------- //

    //std::shared_ptr<rls::Solver<rls::CUDA>> ir;
    //using Solver = rls::Solver<rls::CUDA>;
    //using SolverLogger = rls::solver::iterative::Logger;
    //using SolverConfig = rls::solver::iterative::Config;
    //std::shared_ptr<rls::Solver<rls::CUDA>> solver;
    //std::shared_ptr<SolverConfig> solver_config;

    //// Configure and run solver.
    //{
    //    std::cout << "\n\n";
    //    std::cout << "        matrix: " << matrix << '\n';
    //    std::cout << "sampling_coeff: " << sampling_coeff << '\n';
    //    std::cout << "     tolerance: " << tol << '\n';
    //    auto input_true_sol = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //    using Lsqr = rls::solver::Lsqr
    //        <rls::CUDA, vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
    //    using LsqrConfig = rls::solver::iterative::LsqrConfig
    //        <vtype_solver, vtype_solver_internal, vtype_precond_apply, double, magma_int_t>;
    //    solver_config = LsqrConfig::create();
    //    solver_config->set_precond(precond);
    //    solver_config->set_tolerance(tol);
    //    solver_config->set_stagnation_tolerance(1e-3);  // <= sqrt(gemv precision)
    //    solver_config->set_restarts(restarts);
    //    solver_config->set_min_iterations(miniter);
    //    solver_config->set_iterations(maxiter);
    //    solver_config->set_filename_relres(filename_relres);
    //    solver_config->set_filename_true_error(input_true_sol, filename_true_error);
    //    solver_config->set_filename_stagnation(filename_stagnation);
    //    sol->zeros();
    //    solver = rls::share(Lsqr::create(context, solver_config, mtx, sol, rhs));
    //    ir = rls::solver::Ir<rls::CUDA, magma_int_t>::create(solver, restarts);
    //    ir->run();
    //    //auto logger = ir->get_logger();
    //    //logger.write_history();
    //}
    //std::cout << ">>>exit\n";
}

struct solver_config_t
{
    double tol;
    int maxiter;
    int miniter;
    int restarts;
    double sampling_coeff;
    std::string matrix;
    std::string filename_relres;
    std::string filename_true_error;
    std::string filename_noisy_error;
    std::string filename_stagnation;
    std::string filename_matrix;
    std::string filename_rhs;
    std::string filename_xtrue;
    std::string filename_xnoisy;
    std::string filename_true_similarity;
    std::string filename_noisy_similarity;
};

void run_sparse_lsqr_for_precisions(solver_config_t& parameters)
{

    // reference run
    {
        std::cout << "\n\n  ----> INSTANCE 0 <---- \n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = double;
        using vtype_solver_internal = double;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_0.mtx";
        auto filename_true_error = parameters.filename_true_error + "_0.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_0.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_0.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        run_sparse_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_matrix,
                filename_rhs,
                filename_xtrue
            );
    }

    //{
    //    std::cout << "\n\nINSTANCE 1\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = double;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_1.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_1.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_1.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_sparse_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 2\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = __half;
    //    using vtype_solver_internal = double;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_2.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_2.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_2.mtx";
    //    auto filename_true_similarity = parameters.filename_true_similarity + "_2.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_sparse_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 3\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = double;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_3.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_3.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_3.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_sparse_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 4\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_4.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_4.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_4.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_sparse_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 5\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = __half;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_5.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_5.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_5.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_sparse_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    //@error here.
    //    std::cout << "\n\nINSTANCE 6\n\n";
    //    using vtype_precond_apply = float;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = float;
    //    auto filename_relres = parameters.filename_relres + "_6.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_6.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_6.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_sparse_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 7\n\n";
    //    using vtype_precond_apply = float;
    //    using vtype_precond_internal = __half;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = float;
    //    auto filename_relres = parameters.filename_relres + "_7.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_7.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_7.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_sparse_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}
}

void run_dense_lsqr_for_precisions(solver_config_t& parameters)
{

    //// reference run
    //{
    //    std::cout << "\n\n  ----> INSTANCE 0 <---- \n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = double;
    //    using vtype_solver_internal = double;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_0.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_0.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_0.mtx";
    //    auto filename_true_similarity = parameters.filename_true_similarity + "_0.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_dense_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 1\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = double;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_1.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_1.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_1.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_dense_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 2\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = __half;
    //    using vtype_solver_internal = double;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_2.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_2.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_2.mtx";
    //    auto filename_true_similarity = parameters.filename_true_similarity + "_2.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_dense_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 3\n\n";
    //    using vtype_precond_apply = float;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = double; // if this is double it breaks
    //    auto filename_relres = parameters.filename_relres + "_3.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_3.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_3.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_dense_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    {
        std::cout << "\n\nINSTANCE 4\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = double;
        using vtype_solver_internal = float;
        using vtype_solver = float;
        auto filename_relres = parameters.filename_relres + "_4.mtx";
        auto filename_true_error = parameters.filename_true_error + "_4.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_4.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        run_dense_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_matrix,
                filename_rhs,
                filename_xtrue
            );
    }

    //{
    //    std::cout << "\n\nINSTANCE 5\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = __half;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_5.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_5.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_5.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_dense_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    //@error here.
    //    std::cout << "\n\nINSTANCE 6\n\n";
    //    using vtype_precond_apply = float;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = float;
    //    auto filename_relres = parameters.filename_relres + "_6.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_6.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_6.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_dense_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}

    //{
    //    std::cout << "\n\nINSTANCE 7\n\n";
    //    using vtype_precond_apply = float;
    //    using vtype_precond_internal = __half;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = float;
    //    auto filename_relres = parameters.filename_relres + "_7.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_7.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_7.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    run_dense_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue
    //        );
    //}
}

void run_dense_reg_lsqr_for_precisions(solver_config_t& parameters)
{

    // reference run
    {
        std::cout << "\n\nINSTANCE 0\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = double;
        using vtype_solver_internal = double;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_0.mtx";
        auto filename_true_error = parameters.filename_true_error + "_0.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_0.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_0.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_0.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_0.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    {
        std::cout << "\n\nINSTANCE 1\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = float;
        using vtype_solver_internal = double;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_1.mtx";
        auto filename_true_error = parameters.filename_true_error + "_1.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_1.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_1.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_1.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_1.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    {
        std::cout << "\n\nINSTANCE 2\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = __half;
        using vtype_solver_internal = double;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_2.mtx";
        auto filename_true_error = parameters.filename_true_error + "_2.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_2.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_2.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_2.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_2.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    {
        std::cout << "\n\nINSTANCE 3\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = double;
        using vtype_solver_internal = float;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_3.mtx";
        auto filename_true_error = parameters.filename_true_error + "_3.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_3.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_3.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_3.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_3.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    {
        std::cout << "\n\nINSTANCE 4\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = float;
        using vtype_solver_internal = float;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_4.mtx";
        auto filename_true_error = parameters.filename_true_error + "_4.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_4.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_4.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_4.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_4.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    {
        std::cout << "\n\nINSTANCE 5\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = __half;
        using vtype_solver_internal = float;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_5.mtx";
        auto filename_true_error = parameters.filename_true_error + "_5.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_5.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_5.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_5.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_5.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    {
        //@error here.
        std::cout << "\n\nINSTANCE 6\n\n";
        using vtype_precond_apply = float;
        using vtype_precond_internal = float;
        using vtype_solver_internal = float;
        using vtype_solver = float;
        auto filename_relres = parameters.filename_relres + "_6.mtx";
        auto filename_true_error = parameters.filename_true_error + "_6.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_6.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_6.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_6.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_6.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    {
        std::cout << "\n\nINSTANCE 7\n\n";
        using vtype_precond_apply = float;
        using vtype_precond_internal = __half;
        using vtype_solver_internal = float;
        using vtype_solver = float;
        auto filename_relres = parameters.filename_relres + "_7.mtx";
        auto filename_true_error = parameters.filename_true_error + "_7.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_7.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_7.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_7.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_7.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_dense_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }
}

void run_sparse_reg_lsqr_for_precisions(solver_config_t& parameters)
{

    //// reference run
    //{
    //    std::cout << "\n\nINSTANCE 0\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = double;
    //    using vtype_solver_internal = double;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_0.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_0.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_0.mtx";
    //    auto filename_noisy_error = parameters.filename_noisy_error + "_0.mtx";
    //    auto filename_true_similarity = parameters.filename_true_similarity + "_0.mtx";
    //    auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_0.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    auto filename_xnoisy = parameters.filename_xnoisy;
    //    run_sparse_reg_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_noisy_error,
    //            filename_true_similarity,
    //            filename_noisy_similarity,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue,
    //            filename_xnoisy
    //        );
    //}

    //{
    //    cudaDeviceSynchronize();
    //    std::cout << "\n\nINSTANCE 1\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = double;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_1.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_1.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_1.mtx";
    //    auto filename_noisy_error = parameters.filename_noisy_error + "_1.mtx";
    //    auto filename_true_similarity = parameters.filename_true_similarity + "_1.mtx";
    //    auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_1.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    auto filename_xnoisy = parameters.filename_xnoisy;
    //    run_sparse_reg_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_noisy_error,
    //            filename_true_similarity,
    //            filename_noisy_similarity,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue,
    //            filename_xnoisy
    //        );
    //}

    ////{
    ////    std::cout << "\n\nINSTANCE 2\n\n";
    ////    using vtype_precond_apply = double;
    ////    using vtype_precond_internal = __half;
    ////    using vtype_solver_internal = double;
    ////    using vtype_solver = double;
    ////    auto filename_relres = parameters.filename_relres + "_2.mtx";
    ////    auto filename_true_error = parameters.filename_true_error + "_2.mtx";
    ////    auto filename_stagnation = parameters.filename_stagnation + "_2.mtx";
    ////    auto filename_noisy_error = parameters.filename_noisy_error + "_2.mtx";
    ////    auto filename_true_similarity = parameters.filename_true_similarity + "_2.mtx";
    ////    auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_2.mtx";
    ////    auto filename_matrix = parameters.filename_matrix;
    ////    auto filename_rhs = parameters.filename_rhs;
    ////    auto filename_xtrue = parameters.filename_xtrue;
    ////    auto filename_xnoisy = parameters.filename_xnoisy;
    ////    run_sparse_reg_lsqr_instance
    ////        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    ////        (
    ////            parameters.tol,
    ////            parameters.maxiter,
    ////            parameters.miniter,
    ////            parameters.restarts,
    ////            parameters.sampling_coeff,
    ////            parameters.matrix,
    ////            filename_relres,
    ////            filename_true_error,
    ////            filename_stagnation,
    ////            filename_noisy_error,
    ////            filename_true_similarity,
    ////            filename_noisy_similarity,
    ////            filename_matrix,
    ////            filename_rhs,
    ////            filename_xtrue,
    ////            filename_xnoisy
    ////        );
    ////}

    //{
    //    cudaDeviceSynchronize();
    //    std::cout << "\n\nINSTANCE 3\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = double;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_3.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_3.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_3.mtx";
    //    auto filename_noisy_error = parameters.filename_noisy_error + "_3.mtx";
    //    auto filename_true_similarity = parameters.filename_true_similarity + "_3.mtx";
    //    auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_3.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    auto filename_xnoisy = parameters.filename_xnoisy;
    //    run_sparse_reg_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_noisy_error,
    //            filename_true_similarity,
    //            filename_noisy_similarity,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue,
    //            filename_xnoisy
    //        );
    //}

    //{
    //    cudaDeviceSynchronize();
    //    std::cout << "\n\nINSTANCE 4\n\n";
    //    using vtype_precond_apply = double;
    //    using vtype_precond_internal = float;
    //    using vtype_solver_internal = float;
    //    using vtype_solver = double;
    //    auto filename_relres = parameters.filename_relres + "_4.mtx";
    //    auto filename_true_error = parameters.filename_true_error + "_4.mtx";
    //    auto filename_stagnation = parameters.filename_stagnation + "_4.mtx";
    //    auto filename_noisy_error = parameters.filename_noisy_error + "_4.mtx";
    //    auto filename_true_similarity = parameters.filename_true_similarity + "_4.mtx";
    //    auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_4.mtx";
    //    auto filename_matrix = parameters.filename_matrix;
    //    auto filename_rhs = parameters.filename_rhs;
    //    auto filename_xtrue = parameters.filename_xtrue;
    //    auto filename_xnoisy = parameters.filename_xnoisy;
    //    run_sparse_reg_lsqr_instance
    //        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    //        (
    //            parameters.tol,
    //            parameters.maxiter,
    //            parameters.miniter,
    //            parameters.restarts,
    //            parameters.sampling_coeff,
    //            parameters.matrix,
    //            filename_relres,
    //            filename_true_error,
    //            filename_stagnation,
    //            filename_noisy_error,
    //            filename_true_similarity,
    //            filename_noisy_similarity,
    //            filename_matrix,
    //            filename_rhs,
    //            filename_xtrue,
    //            filename_xnoisy
    //        );
    //}

    ////{
    ////    std::cout << "\n\nINSTANCE 5\n\n";
    ////    using vtype_precond_apply = double;
    ////    using vtype_precond_internal = __half;
    ////    using vtype_solver_internal = float;
    ////    using vtype_solver = double;
    ////    auto filename_relres = parameters.filename_relres + "_5.mtx";
    ////    auto filename_true_error = parameters.filename_true_error + "_5.mtx";
    ////    auto filename_stagnation = parameters.filename_stagnation + "_5.mtx";
    ////    auto filename_noisy_error = parameters.filename_noisy_error + "_5.mtx";
    ////    auto filename_true_similarity = parameters.filename_true_similarity + "_5.mtx";
    ////    auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_5.mtx";
    ////    auto filename_matrix = parameters.filename_matrix;
    ////    auto filename_rhs = parameters.filename_rhs;
    ////    auto filename_xtrue = parameters.filename_xtrue;
    ////    auto filename_xnoisy = parameters.filename_xnoisy;
    ////    run_sparse_reg_lsqr_instance
    ////        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    ////        (
    ////            parameters.tol,
    ////            parameters.maxiter,
    ////            parameters.miniter,
    ////            parameters.restarts,
    ////            parameters.sampling_coeff,
    ////            parameters.matrix,
    ////            filename_relres,
    ////            filename_true_error,
    ////            filename_stagnation,
    ////            filename_noisy_error,
    ////            filename_true_similarity,
    ////            filename_noisy_similarity,
    ////            filename_matrix,
    ////            filename_rhs,
    ////            filename_xtrue,
    ////            filename_xnoisy
    ////        );
    ////}


    std::cout << "\n\n\n\n\n\n\n";
    std::cout << "INSTANCE 6\n";
    {
        cudaDeviceSynchronize();
        //@error here.
        std::cout << "\n\nINSTANCE 6\n\n";
        using vtype_precond_apply = float;
        using vtype_precond_internal = float;
        using vtype_solver_internal = float;
        using vtype_solver = float;
        auto filename_relres = parameters.filename_relres + "_6.mtx";
        auto filename_true_error = parameters.filename_true_error + "_6.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_6.mtx";
        auto filename_noisy_error = parameters.filename_noisy_error + "_6.mtx";
        auto filename_true_similarity = parameters.filename_true_similarity + "_6.mtx";
        auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_6.mtx";
        auto filename_matrix = parameters.filename_matrix;
        auto filename_rhs = parameters.filename_rhs;
        auto filename_xtrue = parameters.filename_xtrue;
        auto filename_xnoisy = parameters.filename_xnoisy;
        run_sparse_reg_lsqr_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation,
                filename_noisy_error,
                filename_true_similarity,
                filename_noisy_similarity,
                filename_matrix,
                filename_rhs,
                filename_xtrue,
                filename_xnoisy
            );
    }

    ////{
    ////    std::cout << "\n\nINSTANCE 7\n\n";
    ////    using vtype_precond_apply = float;
    ////    using vtype_precond_internal = __half;
    ////    using vtype_solver_internal = float;
    ////    using vtype_solver = float;
    ////    auto filename_relres = parameters.filename_relres + "_7.mtx";
    ////    auto filename_true_error = parameters.filename_true_error + "_7.mtx";
    ////    auto filename_stagnation = parameters.filename_stagnation + "_7.mtx";
    ////    auto filename_noisy_error = parameters.filename_noisy_error + "_7.mtx";
    ////    auto filename_true_similarity = parameters.filename_true_similarity + "_7.mtx";
    ////    auto filename_noisy_similarity = parameters.filename_noisy_similarity + "_7.mtx";
    ////    auto filename_matrix = parameters.filename_matrix;
    ////    auto filename_rhs = parameters.filename_rhs;
    ////    auto filename_xtrue = parameters.filename_xtrue;
    ////    auto filename_xnoisy = parameters.filename_xnoisy;
    ////    run_sparse_reg_lsqr_instance
    ////        <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
    ////        (
    ////            parameters.tol,
    ////            parameters.maxiter,
    ////            parameters.miniter,
    ////            parameters.restarts,
    ////            parameters.sampling_coeff,
    ////            parameters.matrix,
    ////            filename_relres,
    ////            filename_true_error,
    ////            filename_stagnation,
    ////            filename_noisy_error,
    ////            filename_true_similarity,
    ////            filename_noisy_similarity,
    ////            filename_matrix,
    ////            filename_rhs,
    ////            filename_xtrue,
    ////            filename_xnoisy
    ////        );
    ////}
}

void run_fgmres_for_precisions(solver_config_t& parameters)
{

    // reference run
    {
        std::cout << "\n\nINSTANCE 0\n\n";
        using vtype_precond_apply = double;
        using vtype_precond_internal = double;
        using vtype_solver_internal = double;
        using vtype_solver = double;
        auto filename_relres = parameters.filename_relres + "_0.mtx";
        auto filename_true_error = parameters.filename_true_error + "_0.mtx";
        auto filename_stagnation = parameters.filename_stagnation + "_0.mtx";
        run_fgmres_instance
            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
            (
                parameters.tol,
                parameters.maxiter,
                parameters.miniter,
                parameters.restarts,
                parameters.sampling_coeff,
                parameters.matrix,
                filename_relres,
                filename_true_error,
                filename_stagnation
            );
    }
//
//    {
//        std::cout << "\n\nINSTANCE 1\n\n";
//        using vtype_precond_apply = double;
//        using vtype_precond_internal = float;
//        using vtype_solver_internal = double;
//        using vtype_solver = double;
//        auto filename_relres = parameters.filename_relres + "_1.mtx";
//        auto filename_true_error = parameters.filename_true_error + "_1.mtx";
//        auto filename_stagnation = parameters.filename_stagnation + "_1.mtx";
//        run_sparse_reg_lsqr_instance
//            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
//            (
//                parameters.tol,
//                parameters.maxiter,
//                parameters.miniter,
//                parameters.restarts,
//                parameters.sampling_coeff,
//                parameters.matrix,
//                filename_relres,
//                filename_true_error,
//                filename_stagnation
//            );
//    }
//
//    {
//        std::cout << "\n\nINSTANCE 2\n\n";
//        using vtype_precond_apply = double;
//        using vtype_precond_internal = __half;
//        using vtype_solver_internal = double;
//        using vtype_solver = double;
//        auto filename_relres = parameters.filename_relres + "_2.mtx";
//        auto filename_true_error = parameters.filename_true_error + "_2.mtx";
//        auto filename_stagnation = parameters.filename_stagnation + "_2.mtx";
//        run_sparse_reg_lsqr_instance
//            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
//            (
//                parameters.tol,
//                parameters.maxiter,
//                parameters.miniter,
//                parameters.restarts,
//                parameters.sampling_coeff,
//                parameters.matrix,
//                filename_relres,
//                filename_true_error,
//                filename_stagnation
//            );
//    }
//
//    {
//        std::cout << "\n\nINSTANCE 3\n\n";
//        using vtype_precond_apply = double;
//        using vtype_precond_internal = double;
//        using vtype_solver_internal = float;
//        using vtype_solver = double;
//        auto filename_relres = parameters.filename_relres + "_3.mtx";
//        auto filename_true_error = parameters.filename_true_error + "_3.mtx";
//        auto filename_stagnation = parameters.filename_stagnation + "_3.mtx";
//        run_sparse_reg_lsqr_instance
//            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
//            (
//                parameters.tol,
//                parameters.maxiter,
//                parameters.miniter,
//                parameters.restarts,
//                parameters.sampling_coeff,
//                parameters.matrix,
//                filename_relres,
//                filename_true_error,
//                filename_stagnation
//            );
//    }
//
//    {
//        std::cout << "\n\nINSTANCE 4\n\n";
//        using vtype_precond_apply = double;
//        using vtype_precond_internal = float;
//        using vtype_solver_internal = float;
//        using vtype_solver = double;
//        auto filename_relres = parameters.filename_relres + "_4.mtx";
//        auto filename_true_error = parameters.filename_true_error + "_4.mtx";
//        auto filename_stagnation = parameters.filename_stagnation + "_4.mtx";
//        run_sparse_reg_lsqr_instance
//            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
//            (
//                parameters.tol,
//                parameters.maxiter,
//                parameters.miniter,
//                parameters.restarts,
//                parameters.sampling_coeff,
//                parameters.matrix,
//                filename_relres,
//                filename_true_error,
//                filename_stagnation
//            );
//    }
//
//    {
//        std::cout << "\n\nINSTANCE 5\n\n";
//        using vtype_precond_apply = double;
//        using vtype_precond_internal = __half;
//        using vtype_solver_internal = float;
//        using vtype_solver = double;
//        auto filename_relres = parameters.filename_relres + "_5.mtx";
//        auto filename_true_error = parameters.filename_true_error + "_5.mtx";
//        auto filename_stagnation = parameters.filename_stagnation + "_5.mtx";
//        run_sparse_reg_lsqr_instance
//            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
//            (
//                parameters.tol,
//                parameters.maxiter,
//                parameters.miniter,
//                parameters.restarts,
//                parameters.sampling_coeff,
//                parameters.matrix,
//                filename_relres,
//                filename_true_error,
//                filename_stagnation
//            );
//    }
//
//    {
//        //@error here.
//        std::cout << "\n\nINSTANCE 6\n\n";
//        using vtype_precond_apply = float;
//        using vtype_precond_internal = float;
//        using vtype_solver_internal = float;
//        using vtype_solver = float;
//        auto filename_relres = parameters.filename_relres + "_6.mtx";
//        auto filename_true_error = parameters.filename_true_error + "_6.mtx";
//        auto filename_stagnation = parameters.filename_stagnation + "_6.mtx";
//        run_sparse_reg_lsqr_instance
//            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
//            (
//                parameters.tol,
//                parameters.maxiter,
//                parameters.miniter,
//                parameters.restarts,
//                parameters.sampling_coeff,
//                parameters.matrix,
//                filename_relres,
//                filename_true_error,
//                filename_stagnation
//            );
//    }
//
//    {
//        std::cout << "\n\nINSTANCE 7\n\n";
//        using vtype_precond_apply = float;
//        using vtype_precond_internal = __half;
//        using vtype_solver_internal = float;
//        using vtype_solver = float;
//        auto filename_relres = parameters.filename_relres + "_7.mtx";
//        auto filename_true_error = parameters.filename_true_error + "_7.mtx";
//        auto filename_stagnation = parameters.filename_stagnation + "_7.mtx";
//        run_sparse_reg_lsqr_instance
//            <vtype_precond_apply, vtype_precond_internal, vtype_solver_internal, vtype_solver>
//            (
//                parameters.tol,
//                parameters.maxiter,
//                parameters.miniter,
//                parameters.restarts,
//                parameters.sampling_coeff,
//                parameters.matrix,
//                filename_relres,
//                filename_true_error,
//                filename_stagnation
//            );
//    }
}


struct instance_config_t
{
    std::vector<std::string> matrix_names;
    std::vector<double> sampling_coeffs;
    std::vector<double> tolerances;
    int warmup_iters;
    int runtime_iters;
};


int main(int argc, char* argv[])
{
    // Instance parameters.
    instance_config_t instance_config;
    instance_config.matrix_names = {"sprand_03"};
    //instance_config.matrix_names = {"hgdp_1_6e4_e3_e6"};
    //instance_config.matrix_names = {"hgdp_1_6e4_e3_e6_n"};
    //instance_config.matrix_names = {"randsvd_1e4_3e2_1e6_3"};
    //instance_config.matrix_names = {"randsvd0"};
    //instance_config.matrix_names = {"randsvd1"};
    //instance_config.sampling_coeffs = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    //instance_config.sampling_coeffs = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0};

    //instance_config.sampling_coeffs = {1.5};
    //instance_config.sampling_coeffs = {2.0};
    //instance_config.sampling_coeffs = {2.5};
    //instance_config.sampling_coeffs = {3.0};
    //instance_config.sampling_coeffs = {1.0};
    instance_config.sampling_coeffs = {4.0};

    //instance_config.tolerances = {1e-8, 1e-15};
    //instance_config.tolerances = {1e-15};
    instance_config.tolerances = {1e-10};
    instance_config.warmup_iters = 3;
    instance_config.runtime_iters = 3;
    std::string filename_base = "../lsqr_output/";

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            parameters.tol = tol;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 20;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtruereg.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg.mtx";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg4.mtx";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            parameters.tol = 1e-8;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 20;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg1/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres1_history";
    //            parameters.filename_true_error = filename_instance + "_true_error1_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error1_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation1_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg1.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg1.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            parameters.tol = 1e-3;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 40;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg2/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres2_history";
    //            parameters.filename_true_error = filename_instance + "_true_error2_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error2_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation2_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg2.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg2.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            parameters.tol = tol;
    //            parameters.maxiter = 300;
    //            parameters.miniter = 20;
    //            parameters.restarts = 20;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            run_fgmres_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            parameters.tol = 1e-6;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 20;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg3/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres3_history";
    //            parameters.filename_true_error = filename_instance + "_true_error3_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error3_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation3_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg3.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg3.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            parameters.tol = 1e-4;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 20;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg4/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres4_history";
    //            parameters.filename_true_error = filename_instance + "_true_error4_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error4_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation4_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg4.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg4.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}


    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            parameters.tol = tol;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 10;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg_n/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtruereg.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg.mtx";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg4.mtx";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            //parameters.tol = tol;
    //            parameters.tol = 1e-8;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 10;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg1_n/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres1_history";
    //            parameters.filename_true_error = filename_instance + "_true_error1_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error1_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation1_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg1.mtx";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg1.mtx";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg1.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    for (auto& matrix : instance_config.matrix_names) {
        for (auto& tol : instance_config.tolerances) {
            for (auto& sampling_coeff : instance_config.sampling_coeffs) {
                solver_config_t parameters;
                //parameters.tol = tol;
                parameters.tol = 1e-5;
                parameters.maxiter = 2;
                parameters.miniter = 2;
                parameters.restarts = 1;
                parameters.sampling_coeff = sampling_coeff;
                parameters.matrix = matrix;
                std::ostringstream streamObj;
                std::ostringstream streamObj2;
                streamObj << parameters.tol;
                streamObj2 << parameters.sampling_coeff;
                streamObj2 << std::fixed;
                streamObj2 << std::setprecision(2);
                std::string filename_instance = filename_base + parameters.matrix + "_reg3_gaussian/" + streamObj2.str() + "_" + streamObj.str();
                parameters.filename_relres = filename_instance + "_relres_history";
                parameters.filename_true_error = filename_instance + "_true_error_history";
                parameters.filename_noisy_error = filename_instance + "_noisy_error_history";
                parameters.filename_stagnation = filename_instance + "_stagnation_history";
                parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
                parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
                parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg3.mtx";
                //parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
                //parameters.filename_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
                parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
                parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg3.mtx";
                run_sparse_reg_lsqr_for_precisions(parameters);
            }
        }
    }

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            //parameters.tol = tol;
    //            parameters.tol = 1e-10;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 5;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            //std::string filename_instance = filename_base + parameters.matrix + "_reg3_n3/" + streamObj2.str() + "_" + streamObj.str();
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg5/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_true_similarity = filename_instance + "_true_similarity_history";
    //            parameters.filename_noisy_similarity = filename_instance + "_noisy_similarity_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg5.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg5.mtx";
    //            run_sparse_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            //parameters.tol = tol;
    //            parameters.tol = 1e-15;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 2;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamobj;
    //            std::ostringstream streamobj2;
    //            streamobj << parameters.tol;
    //            streamobj2 << parameters.sampling_coeff;
    //            streamobj2 << std::fixed;
    //            streamobj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "/" + streamobj2.str() + "_" + streamobj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            run_sparse_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            //parameters.tol = tol;
    //            parameters.tol = 1e-15;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 20;
    //            parameters.restarts = 5;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamobj;
    //            std::ostringstream streamobj2;
    //            streamobj << parameters.tol;
    //            streamobj2 << parameters.sampling_coeff;
    //            streamobj2 << std::fixed;
    //            streamobj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "/" + streamobj2.str() + "_" + streamobj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/hgdp_1_6e4_e3_e6_xtrue.mtx";
    //            run_dense_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            //parameters.tol = tol;
    //            parameters.tol = 1e-10;
    //            //parameters.maxiter = 1000;
    //            //parameters.miniter = 20;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 30;
    //            parameters.restarts = 5;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamobj;
    //            std::ostringstream streamobj2;
    //            streamobj << parameters.tol;
    //            streamobj2 << parameters.sampling_coeff;
    //            streamobj2 << std::fixed;
    //            streamobj2 << std::setprecision(2);
    //            std::string filename_instance = filename_base + parameters.matrix + "/" + streamobj2.str() + "_" + streamobj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_rhs.mtx";
    //            std::cout << "parameters.filename_rhs: " << parameters.filename_rhs << "\n";
    //            //parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            std::cout << "parameters.filename_xtrue: " << parameters.filename_xtrue << '\n';
    //            //parameters.filename_xtrue = "../lsqr_data/hgdp_1_6e4_e3_e6_xtrue.mtx";
    //            run_dense_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            //parameters.tol = tol;
    //            parameters.tol = 3e-5;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 30;
    //            parameters.restarts = 3;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            //std::string filename_instance = filename_base + parameters.matrix + "_reg3_n3/" + streamObj2.str() + "_" + streamObj.str();
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_true_similarity = filename_instance + "_true_similarity_history";
    //            parameters.filename_noisy_similarity = filename_instance + "_noisy_similarity_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            //parameters.filename_xtrue = "../lsqr_data/hgdp_1_6e4_e3_e6_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg.mtx";
    //            run_dense_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    //for (auto& matrix : instance_config.matrix_names) {
    //    for (auto& tol : instance_config.tolerances) {
    //        for (auto& sampling_coeff : instance_config.sampling_coeffs) {
    //            solver_config_t parameters;
    //            //parameters.tol = tol;
    //            parameters.tol = 1e-6;
    //            parameters.maxiter = 1000;
    //            parameters.miniter = 30;
    //            parameters.restarts = 5;
    //            parameters.sampling_coeff = sampling_coeff;
    //            parameters.matrix = matrix;
    //            std::ostringstream streamObj;
    //            std::ostringstream streamObj2;
    //            streamObj << parameters.tol;
    //            streamObj2 << parameters.sampling_coeff;
    //            streamObj2 << std::fixed;
    //            streamObj2 << std::setprecision(2);
    //            //std::string filename_instance = filename_base + parameters.matrix + "_reg3_n3/" + streamObj2.str() + "_" + streamObj.str();
    //            std::string filename_instance = filename_base + parameters.matrix + "_reg1/" + streamObj2.str() + "_" + streamObj.str();
    //            parameters.filename_relres = filename_instance + "_relres_history";
    //            parameters.filename_true_error = filename_instance + "_true_error_history";
    //            parameters.filename_noisy_error = filename_instance + "_noisy_error_history";
    //            parameters.filename_stagnation = filename_instance + "_stagnation_history";
    //            parameters.filename_true_similarity = filename_instance + "_true_similarity_history";
    //            parameters.filename_noisy_similarity = filename_instance + "_noisy_similarity_history";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_matrix = "../lsqr_data/" + matrix + ".mtx";
    //            parameters.filename_rhs = "../lsqr_data/" + matrix + "_breg1.mtx";
    //            parameters.filename_xtrue = "../lsqr_data/" + matrix + "_xtrue.mtx";
    //            //parameters.filename_xtrue = "../lsqr_data/hgdp_1_6e4_e3_e6_xtrue.mtx";
    //            parameters.filename_xnoisy = "../lsqr_data/" + matrix + "_xnoisyreg1.mtx";
    //            run_dense_reg_lsqr_for_precisions(parameters);
    //        }
    //    }
    //}

    return 0;
}
