#ifndef RLS_SOLVER_HPP 
#define RLS_SOLVER_HPP

#include "../matrix/dense/dense.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "../../utils/io.hpp"
#include "magma_v2.h"
#include <iostream>


namespace rls {
namespace solver {
namespace iterative {



class Config {
private:
    double tolerance_    = 1e-8;
    double stagnation_tolerance_ = 1.0;
    double stagnation_weight_ = 1.0;
    double resnorm       = 1.0;
    int max_restarts_    = 1;
    int max_iter_        = 1;
    int min_iter_        = 1;
    bool use_precond_    = false;
    std::shared_ptr<Preconditioner> precond_;
    std::string filename_relres_history_;
    std::string filename_true_error_history_;
    std::string filename_noisy_error_history_;
    std::string filename_stagnation_history_;
    std::string filename_true_sol_;
    std::string filename_noisy_sol_;
    std::string filename_true_sol_similarity_history_;
    std::string filename_noisy_sol_similarity_history_;
    bool record_relres_history_ = false;
    bool record_true_error_history_ = false;
    bool record_noisy_error_history_ = false;
    bool record_stagnation_history_ = false;
    bool record_sol_similarity_history_ = false;

public:
    void set_tolerance(double tolerance)
    {
        tolerance_ = tolerance;
    }

    void set_stagnation_tolerance(double tolerance)
    {
        stagnation_tolerance_ = tolerance;
    }

    void set_stagnation_weight(double weight)
    {
        stagnation_weight_ = weight;
    }

    double get_stagnation_tolerance()
    {
        return stagnation_tolerance_;
    }

    double get_stagnation_weight()
    {
        return stagnation_weight_;
    }

    void set_precond(std::shared_ptr<Preconditioner> precond)
    {
        precond_ = precond;
    }

    void set_iterations(int iterations)
    {
        max_iter_ = iterations;
    }

    void set_restarts(int restarts)
    {
        max_restarts_ = restarts;
    }

    void set_min_iterations(int iterations)
    {
        min_iter_ = iterations;
    }

    int get_min_iterations()
    {
        return min_iter_;
    }

    bool use_precond()
    {
        return use_precond_;
    }

    std::shared_ptr<Preconditioner> get_precond()
    {
        return precond_;
    }

    int get_iterations()
    {
        return max_iter_;
    }

    int get_restarts()
    {
        return max_restarts_;
    }

    double get_tolerance()
    {
        return tolerance_;
    }

    static std::unique_ptr<Config> create()
    {
        return std::unique_ptr<Config>(new Config());
    }

    void set_filename_relres(std::string& s)
    {
        record_relres_history_ = true;
        filename_relres_history_ = s;
    }

    void set_filename_true_error(std::string& st, std::string& s)
    {
        record_true_error_history_ = true;
        filename_true_sol_ = st;
        filename_true_error_history_ = s;
    }

    void set_filename_noisy_error(std::string& st, std::string& s)
    {
        record_noisy_error_history_ = true;
        filename_noisy_sol_ = st;
        filename_noisy_error_history_ = s;
    }

    void set_filename_stagnation(std::string& s)
    {
        record_stagnation_history_ = true;
        filename_stagnation_history_ = s;
    }

    void set_filename_similarity(std::string& s0, std::string& s1)
    {
        record_sol_similarity_history_ = true;
        filename_true_sol_similarity_history_ = s0;
        filename_noisy_sol_similarity_history_ = s1;
    }

    std::string& get_filename_relres()
    {
        return filename_relres_history_;
    }

    std::string& get_filename_true_error()
    {
        return filename_true_error_history_;
    }

    std::string& get_filename_true_sol()
    {
        return filename_true_sol_;
    }

    std::string& get_filename_noisy_error()
    {
        return filename_noisy_error_history_;
    }

    std::string& get_filename_noisy_sol()
    {
        return filename_noisy_sol_;
    }

    std::string& get_filename_stagnation()
    {
        return filename_stagnation_history_;
    }

    std::string& get_filename_true_sol_similarity()
    {
        return filename_true_sol_similarity_history_;
    }

    std::string& get_filename_noisy_sol_similarity()
    {
        return filename_noisy_sol_similarity_history_;
    }

    bool record_relres()
    {
        return record_relres_history_;
    }

    bool record_true_error()
    {
        return record_true_error_history_;
    }

    bool record_noisy_error()
    {
        return record_noisy_error_history_;
    }

    bool record_stagnation()
    {
        return record_stagnation_history_;
    }

    bool record_sol_similarity()
    {
        return record_sol_similarity_history_;
    }

private:
};

class Logger {
public:

    void write_history()
    {
        if (record_relres_history_) {
            auto str = filename_relres_history_.c_str();
            std::cout << "str: " << str << '\n';
            io::write_mtx(str, completed_iterations_, 1,
                (double*)relres_history_->get_values());
        }

        if (record_true_error_history_) {
            auto str = filename_true_error_history_.c_str();
            io::write_mtx(str, completed_iterations_, 1,
                true_error_history_->get_values());
        }

        if (record_stagnation_history_) {
            auto str = filename_stagnation_history_.c_str();
            io::write_mtx(str, completed_iterations_, 1,
                stagnation_history_->get_values());
        }

        if (record_noisy_error_history_) {
            auto str = filename_noisy_error_history_.c_str();
            io::write_mtx(str, completed_iterations_, 1,
                noisy_error_history_->get_values());
        }

        if ((record_true_error_history_) && (record_noisy_error_history_) && (record_sol_similarity_history_)) {
            auto str0 = filename_true_sol_similarity_history_.c_str();
            io::write_mtx(str0, completed_iterations_, 1,
                true_sol_similarity_history_->get_values());

            auto str1 = filename_noisy_sol_similarity_history_.c_str();
            io::write_mtx(str1, completed_iterations_, 1,
                noisy_sol_similarity_history_->get_values());
        }
    }

    static std::unique_ptr<Logger> create(Config* config)
    {
        return std::unique_ptr<Logger>(new Logger(config));
    }

    bool record_relres()
    {
        return record_relres_history_;
    }

    bool record_true_error()
    {
        return record_true_error_history_;
    }

    bool record_noisy_error()
    {
        return record_noisy_error_history_;
    }

    bool record_stagnation()
    {
        return record_stagnation_history_;
    }

    void set_relres_history(int position, double val)
    {
        relres_history_->get_values()[position] = val;
    }

    void set_true_error_history(int position, double val)
    {
        true_error_history_->get_values()[position] = val;
    }

    void set_noisy_error_history(int position, double val)
    {
        noisy_error_history_->get_values()[position] = val;
    }
    void set_stagnation_history(int position, double val)
    {
        stagnation_history_->get_values()[position] = val;
    }

    void set_similarity_history(int position, double val0, double val1)
    {
        true_sol_similarity_history_->get_values()[position] = val0;
        noisy_sol_similarity_history_->get_values()[position] = val1;
    }

    std::string& get_filename_relres()
    {
        return filename_relres_history_;
    }

    std::string& get_filename_true_error()
    {
        return filename_true_error_history_;
    }

    std::string& get_filename_true_sol()
    {
        return filename_true_sol_;
    }

    std::string& get_filename_noisy_sol()
    {
        return filename_noisy_sol_;
    }

    std::string& get_filename_stagnation()
    {
        return filename_stagnation_history_;
    }

    void set_completed_iterations(magma_int_t c)
    {
        completed_iterations_ = c;
    }

private:

    Logger(Config* config)
    {
        total_iterations_ = config->get_restarts() * config->get_iterations();
        record_relres_history_ = config->record_relres();
        record_true_error_history_ = config->record_true_error();
        record_noisy_error_history_ = config->record_noisy_error();
        record_stagnation_history_ = config->record_stagnation();
        record_sol_similarity_history_ = config->record_sol_similarity();
        auto context_cpu = rls::share(rls::Context<rls::CPU>::create());
        if (record_relres_history_) {
            filename_relres_history_ = config->get_filename_relres();
            relres_history_ = rls::matrix::Dense<rls::CPU, double>::create(context_cpu, dim2(total_iterations_, 1));
        }
        if (record_true_error_history_) {
            filename_true_error_history_ = config->get_filename_true_error();
            filename_true_sol_ = config->get_filename_true_sol();
            true_error_history_ = rls::matrix::Dense<rls::CPU, double>::create(context_cpu, dim2(total_iterations_, 1));
        }
        if (record_stagnation_history_) {
            filename_stagnation_history_ = config->get_filename_stagnation();
            stagnation_history_ = rls::matrix::Dense<rls::CPU, double>::create(context_cpu, dim2(total_iterations_, 1));
        }
        if (record_noisy_error_history_) {
            filename_noisy_error_history_ = config->get_filename_noisy_error();
            filename_noisy_sol_ = config->get_filename_noisy_sol();
            noisy_error_history_ = rls::matrix::Dense<rls::CPU, double>::create(context_cpu, dim2(total_iterations_, 1));
        }
        if ((record_true_error_history_) && (record_noisy_error_history_)) {
            filename_true_sol_similarity_history_ = config->get_filename_true_sol_similarity();
            filename_noisy_sol_similarity_history_ = config->get_filename_noisy_sol_similarity();
            true_sol_similarity_history_ = rls::matrix::Dense<rls::CPU, double>::create(context_cpu, dim2(total_iterations_, 1));
            noisy_sol_similarity_history_ = rls::matrix::Dense<rls::CPU, double>::create(context_cpu, dim2(total_iterations_, 1));
        }
    }

    magma_int_t total_iterations_     = 0;
    magma_int_t runs_                 = 1;
    magma_int_t warmup_runs_          = 0;
    magma_int_t max_iterations_       = 0;
    magma_int_t completed_iterations_ = 0;
    magma_int_t completed_iterations_per_restart_ = 0;
    magma_int_t completed_restarts_   = 0;
    magma_int_t global_iterations     = 1;
    double runtime_                   = 0.0;
    double rhsnorm_                   = 0.0;
    double tolerance_                 = 0.0;
    bool measure_runtime_             = false;
    bool new_restart_active           = false;
    bool record_true_error_ = false;
    bool record_sol_similarity_history_;
    std::string filename_relres_history_;
    std::string filename_true_error_history_;
    std::string filename_true_sol_similarity_history_;
    std::string filename_noisy_sol_similarity_history_;
    std::string filename_noisy_error_history_;
    std::string filename_stagnation_history_;
    std::string filename_true_sol_;
    std::string filename_noisy_sol_;
    bool record_relres_history_ = true;
    bool record_true_error_history_ = true;
    bool record_noisy_error_history_ = true;
    bool record_stagnation_history_ = true;
    std::shared_ptr<matrix::Dense<rls::CPU, double>> relres_history_;
    std::shared_ptr<matrix::Dense<rls::CPU, double>> true_error_history_;
    std::shared_ptr<matrix::Dense<rls::CPU, double>> noisy_error_history_;
    std::shared_ptr<matrix::Dense<rls::CPU, double>> stagnation_history_;
    std::shared_ptr<matrix::Dense<rls::CPU, double>> true_sol_similarity_history_;
    std::shared_ptr<matrix::Dense<rls::CPU, double>> noisy_sol_similarity_history_;
};

}

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
    LSQR,
    FGMRES
};


}

template <ContextType device_type>
class Solver {
public:
    virtual void run() = 0;

    std::shared_ptr<Context<device_type>> get_context()
    {
        return context_;
    }

    Solver(std::shared_ptr<Context<device_type>> context)
    {
        context_ = context;
    }

    virtual bool stagnates() = 0;

    virtual bool converges() = 0;

    virtual solver::iterative::Logger get_logger() = 0;

private:
    std::shared_ptr<Context<device_type>> context_;
};

// private:
    // double tolerance_ = 1e-8;
    // magma_int_t max_iter_ = 0;
    // bool use_precond_ = false;
    // SolverValueType combined_value_type_;
    // SolverType type_;
    // double resnorm_ = 1.0;
    // magma_int_t iter_ = 0;
    // logger logger_;

    // void set_logger(logger& logger) {
        // this->logger_ = logger;
    // }

// }   // end of namespace solver

}   // end of namespace rls

#endif
