#ifndef RLS_FGMRES_HPP
#define RLS_FGMRES_HPP


#include <iostream>
#include <memory>
#include <vector>


#include "../include/base_types.hpp"
#include "../preconditioner/gaussian.hpp"
#include "../preconditioner/preconditioner.hpp"


#include "solver.hpp"


namespace rls {
namespace solver {
namespace fgmres {


// Vectors used by Fgmres method.
template <typename value_type_in, typename value_type, typename index_type, ContextType device_type>
struct workspace {
    std::shared_ptr<Context<device_type>> context_;
    // Ccalars.
    index_type inc = 1;
    index_type max_iter_;
    // Vectors and matrices.
    value_type* u;
    value_type* v_basis;
    value_type* w;
    value_type* temp;
    value_type* residual;
    value_type_in* w_in;
    value_type_in* v_in;
    value_type_in* z_in;
    value_type_in* temp_in;
    value_type_in* mtx_in;
    value_type* hessenberg_mtx;
    value_type* hessenberg_mtx_gpu;
    value_type* hessenberg_rhs_gpu;
    value_type* z_basis;
    value_type* hessenberg_rhs;
    std::shared_ptr<std::vector<std::pair<value_type, value_type>>>
        givens_cache;
    std::shared_ptr<matrix::Dense<value_type, CPU>> tmp_cpu;

    workspace(std::shared_ptr<Context<device_type>> context, dim2 size, int max_iter, magma_queue_t& queue)
    {
        std::shared_ptr<Context<CPU>> context_cpu = Context<CPU>::create();
        context_ = context;
        max_iter_ = size[1];
        auto global_len = size[0] + size[1];
        memory::malloc(&u, global_len);
        memory::malloc(&v_basis, global_len * (max_iter_ + 1));
        memory::malloc(&z_basis, global_len * (max_iter_ + 1));
        memory::malloc(&w, global_len);
        memory::malloc(&temp, global_len);
        memory::malloc(&hessenberg_mtx_gpu, max_iter_ * (max_iter_ + 1));
        memory::malloc(&hessenberg_rhs_gpu, (max_iter_ + 1));
        memory::malloc_cpu(&hessenberg_mtx, max_iter_ * (max_iter_ + 1));
        memory::malloc_cpu(&hessenberg_rhs, max_iter_ + 1);
        memory::malloc(&residual, global_len);
        if (!std::is_same<value_type_in, value_type>::value) {
            memory::malloc(&w_in, global_len);
            memory::malloc(&v_in, global_len);
            memory::malloc(&z_in, global_len);
            memory::malloc(&temp_in, global_len);
            memory::malloc(&mtx_in, global_len * size[1]);
        }

        givens_cache = std::shared_ptr<std::vector<std::pair<value_type, value_type>>>(new std::vector<std::pair<value_type, value_type>>);
        givens_cache->resize(static_cast<size_t>(max_iter_ + 1));
        tmp_cpu = matrix::Dense<value_type, CPU>::create(context_cpu, {max_iter_ + 1, 1});
    }

    ~workspace()
    {
        // memory::free(u);
        // memory::free(v_basis);
        // memory::free(z_basis);
        // memory::free(w);
        // memory::free(temp);
        // memory::free(hessenberg_mtx_gpu);
        // memory::free(hessenberg_rhs_gpu);
        // memory::free_cpu(hessenberg_mtx);
        // memory::free_cpu(hessenberg_rhs);
        // memory::free(residual);
        // if (!std::is_same<value_type_in, value_type>::value) {
            // memory::free(w_in);
            // memory::free(v_in);
            // memory::free(z_in);
            // memory::free(mtx_in);
            // memory::free(temp_in);
        // }
    }
};

// Vectors used by Fgmres method.
template <typename value_type_in, typename value_type, typename index_type, ContextType device_type>
struct temp_vectors {
    std::shared_ptr<Context<device_type>> context_;
    index_type max_iter_;
    value_type* u;
    value_type* v_basis;
    value_type* w;
    value_type* temp;
    value_type* residual;
    value_type_in* w_in;
    value_type_in* v_in;
    value_type_in* z_in;
    value_type_in* temp_in;
    value_type_in* mtx_in;
    value_type* hessenberg_mtx;
    value_type* hessenberg_mtx_gpu;
    value_type* hessenberg_rhs_gpu;
    value_type* z_basis;
    index_type inc = 1;
    std::shared_ptr<std::vector<std::pair<value_type, value_type>>>
        givens_cache;
    std::shared_ptr<matrix::Dense<value_type, CPU>> tmp_cpu;
    value_type* hessenberg_rhs;

    temp_vectors(std::shared_ptr<Context<device_type>> context, dim2 size, int max_iter, magma_queue_t& queue)
    {
        std::shared_ptr<Context<CPU>> context_cpu = Context<CPU>::create();
        context_ = context;
        max_iter_ = size[1];
        auto global_len = size[0] + size[1];
        memory::malloc(&u, global_len);
        memory::malloc(&v_basis, global_len * (max_iter_ + 1));
        memory::malloc(&z_basis, global_len * (max_iter_ + 1));
        memory::malloc(&w, global_len);
        memory::malloc(&temp, global_len);
        memory::malloc(&hessenberg_mtx_gpu, max_iter_ * (max_iter_ + 1));
        memory::malloc(&hessenberg_rhs_gpu, (max_iter_ + 1));
        memory::malloc_cpu(&hessenberg_mtx, max_iter_ * (max_iter_ + 1));
        memory::malloc_cpu(&hessenberg_rhs, max_iter_ + 1);
        memory::malloc(&residual, global_len);
        if (!std::is_same<value_type_in, value_type>::value) {
            memory::malloc(&w_in, global_len);
            memory::malloc(&v_in, global_len);
            memory::malloc(&z_in, global_len);
            memory::malloc(&temp_in, global_len);
            memory::malloc(&mtx_in, global_len * size[1]);
        }

        givens_cache = std::shared_ptr<std::vector<std::pair<value_type, value_type>>>(new std::vector<std::pair<value_type, value_type>>);
        givens_cache->resize(static_cast<size_t>(max_iter_ + 1));
        tmp_cpu = matrix::Dense<value_type, CPU>::create(context_cpu, {max_iter_ + 1, 1});
    }

    ~temp_vectors()
    {
        memory::free(u);
        memory::free(v_basis);
        memory::free(z_basis);
        memory::free(w);
        memory::free(temp);
        memory::free(hessenberg_mtx_gpu);
        memory::free(hessenberg_rhs_gpu);
        memory::free_cpu(hessenberg_mtx);
        memory::free_cpu(hessenberg_rhs);
        memory::free(residual);
        if (!std::is_same<value_type_in, value_type>::value) {
            memory::free(w_in);
            memory::free(v_in);
            memory::free(z_in);
            memory::free(mtx_in);
            memory::free(temp_in);
        }
    }
};

// Scalars used by Fgmres method
template <typename value_type, typename index_type>
struct temp_scalars {
    value_type alpha;
    value_type beta;
    value_type rho_bar;
    value_type phi_bar;
    value_type h;
    int* p = nullptr;
};

// Specialized gemv operation for Fgmres on the generalized system
// [I A; A' 0] for input m x n matrix A and m >> n.
template<typename value_type, typename index_type>
void gemv(magma_trans_t trans, index_type num_rows, index_type num_cols,
          value_type alpha, value_type* mtx, index_type ld,
          value_type* u_vector, index_type inc_u, value_type beta,
          value_type* v_vector, index_type inc_v, value_type* tmp,
          magma_queue_t queue)
{
    // Dereference the 2 vector parts.
    auto u0 = u_vector;
    auto u1 = &u_vector[num_rows];
    auto v0 = v_vector;
    auto v1 = &v_vector[num_rows];
    //auto tmp_alpha = -alpha;
    auto tmp_alpha = alpha;

    // Compute first part (indices 0:(m-1))
    blas::copy(num_rows, v0, inc_v, tmp, inc_v, queue);
    blas::copy(num_rows, u0, inc_u, v0, inc_v, queue);
    blas::gemv(MagmaNoTrans, num_rows, num_cols, tmp_alpha, mtx, num_rows, u1,
               inc_u, tmp_alpha, v0, inc_v, queue);
    blas::axpy(num_rows, beta, tmp, inc_v, v0, inc_v, queue);

    // Compute second part (indices m:m+n)
    blas::gemv(MagmaTrans, num_rows, num_cols, -1.0, mtx, num_rows, u0, inc_u,
               beta, v1, inc_v, queue);
}


}  // namespace fgmres


template <typename value_type_in, typename value_type, typename index_type, ContextType device_type>
void run_fgmres(
    matrix::Dense<value_type, device_type>* mtx, matrix::Dense<value_type, device_type>* rhs,
    matrix::Dense<value_type, device_type>* sol,
    preconditioner::preconditioner<value_type, index_type, device_type>*
        precond,
    fgmres::temp_scalars<value_type, index_type>* scalars,
    fgmres::temp_vectors<value_type_in, value_type, index_type, device_type>* vectors,
    magma_int_t max_iter, double tolerance, magma_int_t* iter, double* resnorm,
    magma_queue_t queue, double* t_solve);


template <typename value_type_in, typename value_type, typename index_type, ContextType device_type>
class Fgmres : public generic_solver<device_type> {
public:

    Fgmres(preconditioner::generic_preconditioner<device_type>* precond_in,
           double tolerance_in, std::shared_ptr<Context<device_type>> context)
    {
        this->context_ = context;
        this->precond_ = precond_in;
        this->tolerance_ = tolerance_in;
        use_precond_ = true;
    }

    Fgmres(preconditioner::generic_preconditioner<device_type>* precond_in,
           double tolerance_in, int max_iter_in,
           std::shared_ptr<Context<device_type>> context)
    {
        this->context_ = context;
        this->tolerance_ = tolerance_in;
        this->max_iter_ = max_iter_in;
        this->use_precond_ = true;
    }

    Fgmres(preconditioner::generic_preconditioner<device_type>* precond_in,
           std::shared_ptr<matrix::Dense<value_type, device_type>> mtx,
           std::shared_ptr<matrix::Dense<value_type, device_type>> rhs,
           double tolerance_in) {
        precond_ = precond_in;
        this->mtx_ = mtx;
        this->rhs_ = rhs;
        this->tolerance_ = tolerance_in;
        this->context_ = mtx->get_context();
        this->max_iter_ = mtx_->get_size()[1];
        use_precond_ = true;
        auto num_rows = this->mtx_->get_size()[0];
        auto num_cols = this->mtx_->get_size()[1];
    }

    // Create method (1) of Fgmres solver
    static std::unique_ptr<Fgmres<value_type_in, value_type, index_type, device_type>>
    create(preconditioner::generic_preconditioner<device_type>* precond_in,
           double tolerance_in, std::shared_ptr<Context<device_type>> context)
    {
        return std::unique_ptr<Fgmres<value_type_in, value_type, index_type, device_type>>(
            new Fgmres<value_type_in, value_type, index_type, device_type>(
                precond_in, tolerance_in, context));
    }

    // Create method (2) of Fgmres solver
    static std::unique_ptr<Fgmres<value_type_in, value_type, index_type, device_type>>
    create(preconditioner::generic_preconditioner<device_type>* precond_in,
           double tolerance_in, int max_iter_in,
           std::shared_ptr<Context<device_type>> context)
    {
        return std::unique_ptr<Fgmres<value_type_in, value_type, index_type, device_type>>(
            new Fgmres<value_type_in, value_type, index_type, device_type>(
                precond_in, tolerance_in, max_iter_in, context));
    }


    // Create method (3) of Fgmres solver.
    static std::unique_ptr<Fgmres<value_type_in, value_type, index_type, device_type>>
        create(preconditioner::generic_preconditioner<device_type>* precond_in,
        std::shared_ptr<matrix::Dense<value_type, device_type>> mtx,
        std::shared_ptr<matrix::Dense<value_type, device_type>> rhs,
        double tolerance_in)
    {
        return std::unique_ptr<Fgmres<value_type_in, value_type, index_type, device_type>>(new Fgmres<value_type_in, value_type,
            index_type, device_type>(precond_in, mtx, rhs, tolerance_in));
    }


    void run()
    {
        auto context = this->context_;
        if (use_precond_) {
            index_type iter = 0;
            double resnorm = 1.0;
            for (auto i = 0; i < this->logger_.warmup_runs_; i++) {
                iter = 0;
                resnorm = 1.0;
                glb_sol_->zeros();
                run_fgmres(mtx_.get(), glb_rhs_.get(), glb_sol_.get(),
                    static_cast<preconditioner::preconditioner<value_type, index_type, device_type>*>(precond_), &scalars_, vectors_.get(),
                    this->get_max_iter(), this->get_tolerance(), &iter, &resnorm,
                    this->context_->get_queue(), &this->logger_.runtime_);
            }
            this->logger_.runtime_ = 0.0;
            this->iter_ = 0;
            this->resnorm_ = 0.0;
            for (auto i = 0; i < this->logger_.runs_; i++) {
                iter = 0;
                resnorm = 1.0;
                glb_sol_->zeros();
                run_fgmres(mtx_.get(), glb_rhs_.get(), glb_sol_.get(),
                    static_cast<preconditioner::preconditioner<value_type, index_type, device_type>*>(precond_), &scalars_, vectors_.get(),
                    this->get_max_iter(), this->get_tolerance(), &iter, &resnorm,
                    this->context_->get_queue(), &this->logger_.runtime_);
                this->iter_ += iter;
                this->resnorm_ += resnorm;
            }
            this->logger_.runtime_ = this->logger_.runtime_ / this->logger_.runs_;
            this->iter_ = this->iter_ / this->logger_.runs_;
            this->resnorm_ = this->resnorm_ / this->logger_.runs_;
        } else {
            // Run non-preconditioned FGMRES.
        }
    }

    // Allocates matrices used in fgmres and constructs preconditioner.
    void generate()
    {
        auto queue = this->context_->get_queue();
        // generates rhs and solution vectors
        auto num_rows = this->mtx_->get_size()[0];
        auto num_cols = this->mtx_->get_size()[1];
        auto global_len = num_rows + num_cols;
        sol_ = matrix::Dense<value_type, device_type>::create(this->context_, {num_cols, 1});
        sol_->zeros();
        glb_sol_ = matrix::Dense<value_type, device_type>::create(this->context_, {global_len, 1});
        glb_sol_->zeros();
        glb_rhs_ = matrix::Dense<value_type, device_type>::create(this->context_, {global_len, 1});
        glb_rhs_->zeros();

        auto norm_rhs =
        blas::norm2(num_rows, rhs_->get_values(), 1, queue);
        blas::copy(num_rows, rhs_->get_values(), 1, glb_rhs_->get_values(),
                   1, queue);
        blas::copy(num_cols, sol_->get_values(), 1,
            &glb_rhs_->get_values()[num_rows], 1, queue);
        auto norm_rhs2 = blas::norm2(global_len, glb_rhs_->get_values(), 1,
            queue);
        this->max_iter_ = num_cols;

        vectors_ = std::shared_ptr<
            fgmres::temp_vectors<value_type_in, value_type, index_type, device_type>>(
            new fgmres::temp_vectors<value_type_in, value_type, index_type, device_type>(
                this->context_, mtx_->get_size(), this->max_iter_, queue));

        // Converts matrix to reduced precision.
        utils::convert(this->context_, mtx_->get_size()[0], mtx_->get_size()[1],
            this->mtx_->get_values(), mtx_->get_ld(), vectors_->mtx_in,
            this->mtx_->get_ld());

        if (use_precond_) {
            precond_->generate();
            precond_->compute();
        }

        // auto tmp = std::move(precond_);
    }

private:

    void allocate_vectors(dim2 size);

    void free_vectors();

    bool use_precond_ = false;
    std::shared_ptr<fgmres::temp_vectors<value_type_in, value_type, index_type, device_type>>
        vectors_;
    fgmres::temp_scalars<value_type, index_type> scalars_;
    preconditioner::generic_preconditioner<device_type>* precond_;
    preconditioner::generic_preconditioner<device_type>* precond_in_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> mtx_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> dmtx_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> rhs_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> sol_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> glb_rhs_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> glb_sol_;
};


}  // namespace solver
}  // namespace rls


#endif  // RLS_FGMRES_HPP
