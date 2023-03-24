#ifndef LSQR_HPP
#define LSQR_HPP


#include "../include/base_types.hpp"
#include "solver.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "../preconditioner/gaussian.hpp"

#include <iostream>
#include <memory>


namespace rls {
namespace solver {


template <typename value_type_in, typename value_type, typename index_type>
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
        std::cout << "in constructor of temp_vectors\n";
        std::cout << "size[0]: " << size[0] << '\n';
        std::cout << "size[1]: " << size[1] << '\n';
        memory::malloc(&u, size[0]);
        std::cout << "(u == nullptr): " << (u == nullptr) << '\n';
        memory::malloc(&v, size[0]);
        std::cout << "(v == nullptr): " << (v == nullptr) << '\n';
        memory::malloc(&w, size[0]);
        std::cout << "(w == nullptr): " << (w == nullptr) << '\n';
        memory::malloc(&temp, size[0]);
        std::cout << "(temp == nullptr): " << (temp == nullptr) << '\n';
        if (!std::is_same<value_type_in, value_type>::value) {
            memory::malloc(&u_in, size[0]);
            memory::malloc(&v_in, size[0]);
            memory::malloc(&temp_in, size[0]);
            memory::malloc(&mtx_in, size[0] * size[1]);
        }
    }

    ~temp_vectors() {
        std::cout << "in destructor of temp_vectors\n";
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

template <typename value_type, typename index_type>
struct temp_scalars{
    value_type alpha;
    value_type beta;
    value_type rho_bar;
    value_type phi_bar;
    int* p = nullptr;

    temp_scalars() { std::cout << "in temp_scalars constructor\n"; }
    ~temp_scalars() { std::cout << "in temp_scalars destructor\n"; }
};    

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

// template <typename value_type_in, typename value_type, typename index_type>
// void run_lsqr<value_type_in, value_type, index_type>(matrix::dense<value_type>* mtx,
//     matrix::dense<value_type>* rhs, preconditioner::preconditioner* precond,
    // magma_int_t* iter, double t_solve);       

template <typename value_type_in, typename value_type, typename index_type>
void initialize(dim2 size, value_type* mtx, value_type* rhs,
                value_type* precond_mtx, index_type ld_precond,
                temp_scalars<value_type, index_type>* scalars,
                temp_vectors<value_type_in, value_type, index_type>* vectors,
                magma_queue_t queue, double* t_solve);

template <typename value_type_in, typename value_type, typename index_type>
void run_lsqr(
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
class lsqr : public solver {
public:
    lsqr() {
        this->type = Lsqr;
        std::cout << "in lsqr\n";
        this->set_type();
    }

    lsqr(dim2 size, double tolerance_in, magma_int_t iterations_in, magma_queue_t queue_in) {
        this->type = Lsqr;
        this->set_type();
        // this->queue = queue_in;
        // solver(tolerance_in, iterations_in);
        // allocate_vectors(size);
    }

    lsqr(preconditioner::generic_preconditioner* precond_in,
        std::shared_ptr<matrix::dense<value_type>> rhs_in,
        std::shared_ptr<matrix::dense<value_type>> sol_in,
        double tolerance_in, int max_iter_in, magma_queue_t queue_in) {
        
        this->type = Lsqr;
        switch (precond_in->get_precond_valuetype()) {
            // case preconditioner::FP64_FP64:
            // {
                // this->mtx = static_cast<preconditioner::preconditioner<double, double, int>*>(precond_in)->get_mtx();
                // break;
            // }
            // default:
                // break;
        }
        // mtx = mtx_in;
        rhs = rhs_in;
        sol = sol_in;
        // precond_mtx = precond_in;
        this->tolerance = tolerance_in;
        this->max_iter = max_iter_in;
        this->queue = queue_in;
        use_precond = true;
        this->set_type();
    }

    lsqr(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in, int max_iter_in,
        magma_queue_t queue_in) {
        this->tolerance = tolerance_in;
        this->max_iter = max_iter_in;
        this->type = Lsqr;
        this->queue = queue_in;
        use_precond = true;
        this->set_type();
    }

    lsqr(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in,
        magma_queue_t& queue_in) {
        this->precond = precond_in;
        this->queue = queue_in;
        this->tolerance = tolerance_in;
        this->type = Lsqr;
        use_precond = true;
        this->set_type();
    }

    lsqr(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in,
        std::shared_ptr<Context> context) {
        std::cout << "LSQR constructor\n";
        this->precond = precond_in;
        this->tolerance = tolerance_in;
        this->type = Lsqr;
        this->context = context;
        use_precond = true;
        this->set_type();
    }

    lsqr(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in, int max_iter_in,
        std::shared_ptr<Context> context) {
        this->tolerance = tolerance_in;
        this->max_iter = max_iter_in;
        this->type = Lsqr;
        // this->queue = queue_in;
        this->context = context;
        use_precond = true;
        this->set_type();
    }

    static std::unique_ptr<lsqr<value_type_in, value_type, index_type>>
        create(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in,
        std::shared_ptr<Context> context)
    {
        return std::unique_ptr<lsqr<value_type_in, value_type, index_type>>(new lsqr<value_type_in, value_type, index_type>(precond_in, tolerance_in, context));
    } 

    static std::unique_ptr<lsqr<value_type_in, value_type, index_type>>
        create(preconditioner::generic_preconditioner* precond_in,
        double tolerance_in, int max_iter_in,
        std::shared_ptr<Context> context)
    {
        return std::unique_ptr<lsqr<value_type_in, value_type, index_type>>(new lsqr<value_type_in, value_type, index_type>(precond_in, tolerance_in, max_iter_in, context));
    } 


    void generate(std::string& filename_mtx, std::string& filename_rhs)
    {
        auto context = this->context;
        auto queue = context->get_queue();
        mtx = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());
        sol = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());
        init_sol = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());
        rhs = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>());

        // Initializes matrix and rhs.
        mtx->generate(filename_mtx, queue);
        auto num_rows = mtx->get_size()[0];
        auto num_cols = mtx->get_size()[1];

        // generates rhs and solution vectors
        sol->generate({num_cols, 1});
        sol->zeros();          
        init_sol->generate({num_cols, 1});
        init_sol->zeros();          
        rhs->generate(filename_rhs, queue);

        vectors = std::shared_ptr<temp_vectors<value_type_in, value_type, index_type>>(
           new temp_vectors<value_type_in, value_type, index_type>(mtx->get_size()));

        if (use_precond) {
            auto this_precond = dynamic_cast<preconditioner::gaussian<value_type_in, value_type, index_type>*>(precond);
            this_precond->generate(mtx.get());
            this_precond->compute(mtx.get());
        }

        this->max_iter = num_rows;
    }

    // lsqr(dim2 size, double tolerance_in, magma_int_t iterations_in,
    //     preconditioner::preconditioner<value_type_in, value_type,
    //     index_type>& precond_in, magma_queue_t queue_in) : solver<value_type, index_type>(tolerance_in, iterations_in) {
    //     // this.queue = queue_in;
    //     // use_precond = true;
    //     // precond = std::move(precond);
    //     // allocate_vectors(size);
    // }

    // ~lsqr() {
    //     // free_vectors();
    // }
    

    // void run() { std::cout << "run in lsqr\n"; }
    ~lsqr() {
        std::cout << "in ~lsqr()" << '\n';
    }

    void set_type() {
        if ((typeid(value_type_in) == typeid(double)) && (typeid(value_type_in) == typeid(double)))
        {
            combined_value_type = FP64_FP64;
        }
        else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(double)))
        {
            combined_value_type = FP32_FP64;
        }
        else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(double)))
        {
            combined_value_type = FP16_FP64;
        }
        else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(float)))
        {
            combined_value_type = FP32_FP32;
        }
        else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(float)))
        {
            combined_value_type = FP16_FP32;
        }
    }

    void run();
    // virtual void run() = 0;

    bool check_stopping_criteria(matrix::dense<value_type>& mtx_in, matrix::dense<value_type>& rhs_in,
        matrix::dense<value_type>& sol_in,
        matrix::dense<value_type>& res_vector_in, index_type* iter, double* resnorm);
        
    void step_1(matrix::dense<value_type> mtx,
                temp_scalars<value_type, index_type>* scalars,
                temp_vectors<value_type_in, value_type, index_type>* vectors);

    void step_2(matrix::dense<value_type>& mtx_in,
                matrix::dense<value_type>& sol_in,
                temp_scalars<value_type, index_type>* scalars,
                temp_vectors<value_type_in, value_type, index_type>* vectors);

private:
    void allocate_vectors(dim2 size);
    void free_vectors();

    bool use_precond = false;
    magma_int_t iter;
    double resnorm;
    double t_solve;
    std::shared_ptr<temp_vectors<value_type_in, value_type, index_type>> vectors;
    temp_scalars<value_type, index_type> scalars;
    preconditioner::generic_preconditioner* precond;
    std::shared_ptr<matrix::dense<value_type>> mtx;
    std::shared_ptr<matrix::dense<value_type>> dmtx;
    std::shared_ptr<matrix::dense<value_type>> rhs;
    std::shared_ptr<matrix::dense<value_type>> sol;
    std::shared_ptr<matrix::dense<value_type>> init_sol;
};

template <typename value_type_in, typename value_type, typename index_type>
void lsqr<value_type_in, value_type, index_type>::run()
{
    auto context = this->context;
    if (use_precond) {
        std::cout << "in lsqr::run\n";
        std::cout << "before run\n";
        // run_lsqr(mtx.get(), rhs.get(), sol.get(), static_cast<preconditioner::preconditioner<value_type_in, value_type, index_type>*>(precond),
            // scalars.get(), vectors.get(), this->get_max_iter(), this->get_tolerance(), &iter, &resnorm, this->get_queue(), &t_solve);
        run_lsqr(mtx.get(), rhs.get(), sol.get(), static_cast<preconditioner::preconditioner<value_type_in, value_type, index_type>*>(precond),
            &scalars, vectors.get(), this->get_max_iter(), this->get_tolerance(), &iter, &resnorm, context->get_queue(), &t_solve);
    }
    else {
        // run_solver();
    }

    // iter = 0;
    // auto t = precond->get_precond_mtx;
    // initialize(mtx->get_size(), mtx->get_values(), rhs->get_values(), precond->get_precond_mtx, t_solve);
    // auto size = mtx.get_size();
    // // double t = magma_sync_wtime(this.queue);
    // while (1) {
        // step_1(mtx, scalars, vectors);
    //     step_2(mtx, sol, this.precond, scalars, vectors);
    //     if (check_stopping_criteria(mtx, rhs, sol, vectors.temp,
    //         iter, resnorm)) {
    //         break;
    //     }
    // }
    // // *t_solve += (magma_sync_wtime(this.queue) - t);
}

}
}



#endif  // run_HPP
