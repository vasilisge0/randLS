#ifndef RLS_FGMRES_HPP
#define RLS_FGMRES_HPP


#include <iostream>
#include <memory>


#include "../include/base_types.hpp"
#include "solver.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "../preconditioner/gaussian.hpp"


namespace rls {
namespace solver {
namespace fgmres {


template<typename value_type_in, typename value_type, typename index_type>
struct temp_vectors{
    value_type* u;
    value_type* v;
    value_type* w;
    value_type* temp;
    value_type_in* u_in;
    value_type_in* v_in;
    value_type_in* temp_in;
    value_type_in* mtx_in;
    index_type inc = 1;

    temp_vectors(dim2 size) {
        memory::malloc(&u, size[0]);
        memory::malloc(&v, size[0]);
        memory::malloc(&w, size[0]);
        memory::malloc(&temp, size[0]);
        if (!std::is_same<value_type_in, value_type>::value) {
            memory::malloc(&u_in, size[0]);
            memory::malloc(&v_in, size[0]);
            memory::malloc(&temp_in, size[0]);
            memory::malloc(&mtx_in, size[0] * size[1]);
        }
    }

    ~temp_vectors() {
        memory::free(u);
        memory::free(v);
        memory::free(w);
        memory::free(temp);
        if (!std::is_same<value_type_in, value_type>::value) {
            memory::free(u_in);
            memory::free(v_in);
            memory::free(mtx_in);
            memory::free(temp_in);
        }
    }
};

template<typename value_type_in, typename value_type, typename index_type>
struct temp_scalars{
    value_type alpha;
    value_type beta;
    value_type rho_bar;
    value_type phi_bar;
    int* p = nullptr;

    temp_scalars() {}
    ~temp_scalars() {}
};    


} // namespace fgmres


template <typename value_type, typename index_type>
void run(index_type num_rows, index_type num_cols, value_type* mtx,
          value_type* rhs, value_type* init_sol, value_type* sol,
          index_type max_iter, index_type* iter, value_type tol,
          double* resnorm, magma_queue_t queue);

template <typename value_type_in, typename value_type, typename index_type>
void run(index_type num_rows, index_type num_cols, value_type* mtx,
          value_type* rhs, value_type* init_sol, value_type* sol,
          index_type max_iter, index_type* iter, value_type tol,
          double* resnorm, value_type* precond_mtx, index_type ld_precond,
          magma_queue_t queue, double* t_solve);

template <typename value_type_in, typename value_type, typename index_type>
void initialize(dim2 size, value_type* mtx, value_type* rhs,
                value_type* precond_mtx, index_type ld_precond,
                temp_scalars<value_type, index_type>* scalars,
                temp_vectors<value_type_in, value_type, index_type>* vectors,
                magma_queue_t queue, double* t_solve);

template <typename value_type_in, typename value_type, typename index_type>
void run_fgmres(
    matrix::dense<value_type>* mtx,
    matrix::dense<value_type>* rhs,
    matrix::dense<value_type>* sol,
    preconditioner::preconditioner<value_type_in, value_type, index_type>* precond,
    temp_scalars<value_type, index_type>* scalars,
    temp_vectors<value_type_in, value_type, index_type>* vectors,
    magma_int_t max_iter, double tolerance,
    magma_int_t* iter, double* resnorm, magma_queue_t queue, double* t_solve);


template<typename value_type_in, typename value_type,
         typename index_type>
class Fgmres : public generic_solver {
public:


    Fgmres() {
        this->type_ = FGMRES;
        this->set_type();
    }

    Fgmres(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in,
        std::shared_ptr<Context> context) {
        precond_ = precond_in;
        tolerance_ = tolerance_in;
        type_ = FGMRES;
        context_ = context;
        use_precond_ = true;
        this->set_type();
    }

    Fgmres(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in, int max_iter_in,
        std::shared_ptr<Context> context) {
        tolerance_ = tolerance_in;
        max_iter_ = max_iter_in;
        type_ = FGMRES;
        context_ = context;
        use_precond_ = true;
        this->set_type();
    }

    static std::unique_ptr<Fgmres<value_type_in, value_type, index_type>>
        create(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in,
        std::shared_ptr<Context> context)
    {
        return std::unique_ptr<Fgmres<value_type_in, value_type, index_type>>(new Fgmres<value_type_in, value_type, index_type>(precond_in, tolerance_in, context));
    } 

    static std::unique_ptr<Fgmres<value_type_in, value_type, index_type>>
        create(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in, int max_iter_in,
        std::shared_ptr<Context> context)
    {
        return std::unique_ptr<Fgmres<value_type_in, value_type, index_type>>(new Fgmres<value_type_in, value_type, index_type>(precond_in, tolerance_in, max_iter_in, context));
    } 

    void generate(std::string& filename_mtx, std::string& filename_rhs)
    {
        auto queue = context_->get_queue();
        mtx_ = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());
        sol_ = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());
        init_sol_ = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());
        rhs_ = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());
        auto tmp_rhs = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());

        // Initializes matrix and rhs.
        mtx_->generate(filename_mtx, queue);
        auto num_rows = mtx_->get_size()[0];
        auto num_cols = mtx_->get_size()[1];

        // generates rhs and solution vectors
        sol_->generate({num_rows + num_cols, 1});
        sol_->zeros();          
        tmp_rhs->generate(filename_rhs, queue);
        rhs_->generate({num_rows + num_cols, 1});
        rhs_->zeros();          
        blas::copy(num_rows, tmp_rhs->get_values(), 1, rhs_->get_values(), 1,
            this->context_->get_queue());


        vectors_ = std::shared_ptr<fgmres::temp_vectors<value_type_in, value_type, index_type>>(
           new temp_vectors<value_type_in, value_type, index_type>(mtx_->get_size()));

        if (use_precond_) {
            // auto this_precond = dynamic_cast<preconditioner::gaussian<value_type_in, value_type, index_type>*>(precond_);
            // this_precond->generate(mtx_.get());
            // this_precond->compute(mtx_.get());
        }

        max_iter_ = num_rows;
    }    

    // ~fgmres() { }

    // void set_type() {
    //     if ((typeid(value_type_in) == typeid(double)) && (typeid(value_type_in) == typeid(double)))
    //     {
    //         combined_value_type_ = FP64_FP64;
    //     }
    //     else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(double)))
    //     {
    //         combined_value_type_ = FP32_FP64;
    //     }
    //     else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(double)))
    //     {
    //         combined_value_type_ = FP16_FP64;
    //     }
    //     else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(float)))
    //     {
    //         combined_value_type_ = FP32_FP32;
    //     }
    //     else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(float)))
    //     {
    //         combined_value_type_ = FP16_FP32;
    //     }
    // }

    void run()
    {
        auto context = this->context_;
        if (use_precond_) {
            // run_fgmres(mtx_.get(), rhs_.get(), sol_.get(), static_cast<preconditioner::preconditioner<value_type_in, value_type, index_type>*>(precond_),
                // &scalars_, vectors_.get(), this->get_max_iter(), this->get_tolerance(), &iter_, &resnorm_, context_->get_queue(), &t_solve_);
        }
        else {
            // Run non-preconditioned FGMRES.
        }
    }

private:
    void allocate_vectors(dim2 size);
    void free_vectors();
    bool use_precond_ = false;
    magma_int_t iter_;
    double resnorm_;
    double t_solve_;
    std::shared_ptr<fgmres::temp_vectors<value_type_in, value_type, index_type>> vectors_;
    // temp_scalars<value_type, index_type> scalars_;
    preconditioner::generic_preconditioner* precond_;
    std::shared_ptr<matrix::dense<value_type>> mtx_;
    std::shared_ptr<matrix::dense<value_type>> dmtx_;
    std::shared_ptr<matrix::dense<value_type>> rhs_;
    std::shared_ptr<matrix::dense<value_type>> sol_;
    std::shared_ptr<matrix::dense<value_type>> init_sol_;

};


}   // end of solver namespace
}   // end of fgmres namespace


#endif  // RLS_FGMRES_HPP