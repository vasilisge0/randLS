#ifndef RLS_GAUSSIAN_HPP
#define RLS_GAUSSIAN_HPP


#include <iostream>
#include <memory>
#include "magma_v2.h"


#include "preconditioner.hpp"
#include "../../utils/io.hpp"
#include "../memory/magma_context.hpp"
#include "../dense/dense.hpp"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../blas/blas.hpp"


namespace rls {
namespace preconditioner {


template <typename value_type_internal, typename value_type,
          typename index_type>
struct state{ 
    value_type_internal* dmtx_rp = nullptr;
    value_type_internal* dsketch_rp = nullptr;
    value_type_internal* dresult_rp = nullptr;
    value_type* tau = nullptr;

    void allocate(index_type ld_mtx, index_type num_cols_mtx, 
        index_type num_rows_sketch, index_type num_cols_sketch, index_type ld_sketch,
        index_type ld_r_factor) {
        memory::malloc(&dmtx_rp, ld_mtx * num_cols_mtx);
        memory::malloc(&dsketch_rp, ld_sketch * num_cols_sketch);
        memory::malloc(&dresult_rp, ld_r_factor * num_cols_mtx);
        memory::malloc_cpu(&tau, num_rows_sketch);
    }

    void free() {
        memory::free(dmtx_rp);
        memory::free(dsketch_rp);
        memory::free(dresult_rp);
        memory::free_cpu(tau);
    }
};

template <typename value_type_internal, typename value_type, typename index_type>
void generate_old(index_type num_rows_sketch, index_type num_cols_sketch,
              value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx,
              value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
              index_type ld_r_factor,
              state<value_type_internal, value_type, index_type>* precond_state, 
              std::shared_ptr<Context> context, double* runtime, double* t_mm,
              double* t_qr);


template <typename value_type_internal, typename value_type,
          typename index_type>
void generate_old(index_type num_rows_sketch, index_type num_cols_sketch, value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx, value_type* dmtx, index_type ld_mtx,
              value_type* dr_factor, index_type ld_r_factor,
              value_type* hat_mtx, std::shared_ptr<Context> context);

// template <typename value_type_in, typename value_type,
        //   typename index_type>
// void compute_precond(index_type num_rows_sketch, index_type num_cols_sketch, value_type* dsketch, index_type ld_sketch,
            //   index_type num_rows_mtx, index_type num_cols_mtx, value_type* dmtx, index_type ld_mtx,
            //   value_type* dr_factor, index_type ld_r_factor,
            //   value_type* hat_mtx, detail::magma_info& info);

template <typename value_type_in, typename value_type,
          typename index_type>
void compute_precond(index_type num_rows_sketch, index_type num_cols_sketch,
              value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx,
              value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
              index_type ld_r_factor, value_type* hat_mtx,
              Context& info, double* runtime);


template<typename value_type_in, typename value_type, typename index_type>
class gaussian : public preconditioner<value_type_in, value_type, index_type> {
public:

    gaussian() { 
        this->set_type();
        this->precond_type_ = Gaussian;
        this->sampling_coeff_ = 1.5;
    }

    gaussian(std::shared_ptr<Context> context) { 
        this->set_type();
        this->precond_type_ = Gaussian;
        this->sampling_coeff_ = 1.5;
        this->context_ = context;
    }

    gaussian(matrix::dense<value_type>& sketch_mtx) {
        sketch_mtx_ = std::move(sketch_mtx);
        this->set_type();
    }

    gaussian(std::shared_ptr<matrix::dense<value_type>> mtx,
        std::shared_ptr<matrix::dense<value_type>> sketch_mtx) {

        this->set_type();
        this->mtx_ = mtx;
        this->sketch_mtx_ = sketch_mtx;
        auto size = this->mtx_->get_size();

        auto sampled_rows = sketch_mtx_->get_size()[0];
        precond_mtx_ = std::shared_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>());
        precond_mtx_internal_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        dt_ = std::unique_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>());
        mtx_rp_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        dsketch_rp_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        dresult_rp_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        tau_ = std::unique_ptr<matrix::dense<index_type>>(new matrix::dense<index_type>());

        precond_mtx_->generate({sampled_rows, size[1]});
        precond_mtx_internal_->generate({sampled_rows, size[1]});
        dt_->generate(size);
        mtx_rp_->generate(size);
        dsketch_rp_->generate({sampled_rows, size[0]});
        dresult_rp_->generate({sampled_rows, size[1]});
        tau_->generate_cpu({size[0], 1});
    }

    ~gaussian() {}

    void generate(double* runtime);

    void generate(matrix::dense<value_type>* mtx) {
        auto context = this->context_;
        auto num_rows = mtx->get_size()[0];
        auto num_cols = mtx->get_size()[1];
        index_type sampled_rows = (index_type)(sampling_coeff_ * num_cols);
        auto size = mtx->get_size();

        // Generates sketch matrix.
        sketch_mtx_ = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>({sampled_rows, num_rows}));
        sketch_mtx_->generate();
        if (std::is_same<value_type, double>::value) {
            curandGenerateNormalDouble(
                context->get_generator(), (double*)sketch_mtx_->get_values(), sampled_rows * num_rows, 0, 1);
        }
        else if (std::is_same<value_type, float>::value) {
            curandGenerateNormal(
                context->get_generator(), (float*)sketch_mtx_->get_values(), sampled_rows * num_rows, 0, 1);
        }
        cudaDeviceSynchronize();

        // Construct preconditioner
        precond_mtx_ = std::shared_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>({sampled_rows, size[1]}));
        precond_mtx_internal_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>({sampled_rows, size[1]}));
        dt_ = std::unique_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>(size));
        mtx_rp_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>(size));
        dsketch_rp_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>({sampled_rows, size[0]}));
        dresult_rp_ = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>({sampled_rows, size[1]}));
        tau_ = std::unique_ptr<matrix::dense<index_type>>(new matrix::dense<index_type>({size[0], 1}));

        // Generate matrices.
        precond_mtx_->generate();
        precond_mtx_internal_->generate();
        dt_->generate();
        mtx_rp_->generate();
        dsketch_rp_->generate();
        dresult_rp_->generate();
        tau_->generate_cpu();
    }

    gaussian<value_type_in, value_type, index_type>* get() { return this; }

    matrix::dense<value_type>* get_mtx() { return precond_mtx_.get(); }

    static std::unique_ptr<gaussian<value_type_in, value_type, index_type>> create(std::shared_ptr<Context> context) {
        return std::unique_ptr<gaussian<value_type_in, value_type, index_type>>(new gaussian<value_type_in, value_type, index_type>(context));
    }

    void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u)
    {
        auto queue = this->context_->get_queue();
        auto size = precond_mtx_->get_size();
        blas::trsv(MagmaUpper, trans, MagmaNonUnit, size[1], precond_mtx_->get_values(),
            size[0], u_vector, inc_u, queue);
        magma_queue_sync(queue);
    }

    void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u, magma_queue_t queue)
    {
        auto size = precond_mtx_->get_size();
        blas::trsv(MagmaUpper, trans, MagmaNonUnit, size[1], precond_mtx_->get_values(),
            size[0], u_vector, inc_u, this->context_->get_queue());
        magma_queue_sync(queue);
    }

    void compute(matrix::dense<value_type>* dmtx)
    {
        value_type* hat_mtx = nullptr;
        double runtime;
        auto precond_state = new state<value_type_in, value_type, index_type>();
        double runtime_local;
        double t_mm;
        double t_qr;
        precond_state->allocate(dmtx->get_size()[0], dmtx->get_size()[1], 
            sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1], sketch_mtx_->get_size()[0],
            sketch_mtx_->get_size()[0]);
        generate_old(sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1],
            sketch_mtx_->get_values(), sketch_mtx_->get_size()[0],
            dmtx->get_size()[0], dmtx->get_size()[1],
            dmtx->get_values(), dmtx->get_size()[0], precond_mtx_->get_values(),
            precond_mtx_->get_size()[0],
            precond_state, 
            this->context_, &runtime_local, &t_mm, &t_qr);
        precond_state->free();
    }

    value_type* get_values() { return precond_mtx_->get_values(); }

    dim2 get_size() { return precond_mtx_->get_size(); }

private:
    value_type sampling_coeff_;
    std::shared_ptr<matrix::dense<value_type>> mtx_;
    std::shared_ptr<matrix::dense<value_type>> sketch_mtx_;
    std::shared_ptr<matrix::dense<value_type>> precond_mtx_;
    std::unique_ptr<matrix::dense<value_type_in>> precond_mtx_internal_;
    std::unique_ptr<matrix::dense<value_type>> dt_;
    std::unique_ptr<matrix::dense<value_type_in>> mtx_rp_;
    std::unique_ptr<matrix::dense<value_type_in>> dsketch_rp_;
    std::unique_ptr<matrix::dense<value_type_in>> dresult_rp_;
    std::unique_ptr<matrix::dense<index_type>> tau_;
};


}  // namespace preconditioner
}  // namespace rls


#endif
