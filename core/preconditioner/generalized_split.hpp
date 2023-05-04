#ifndef RLS_GENERALIZED_SPLIT_HPP
#define RLS_GENERALIZED_SPLIT_HPP


#include <iostream>
#include <memory>
#include "magma_v2.h"


#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../cuda/solver/lsqr_kernels.cuh"
#include "../../utils/io.hpp"
#include "../blas/blas.hpp"
#include "../dense/dense.hpp"
#include "../memory/magma_context.hpp"
#include "preconditioner.hpp"


namespace rls {
namespace preconditioner {
namespace generalized_split {

template <typename value_type_in, typename value_type,
          typename index_type>
struct state {
    value_type_in* dmtx_rp = nullptr;
    value_type_in* dsketch_rp = nullptr;
    value_type_in* dresult_rp = nullptr;
    value_type* tau = nullptr;

    void allocate(index_type ld_mtx, index_type num_cols_mtx,
                  index_type num_rows_sketch, index_type num_cols_sketch,
                  index_type ld_sketch, index_type ld_r_factor)
    {
        memory::malloc(&dmtx_rp, ld_mtx * num_cols_mtx);
        memory::malloc(&dsketch_rp, ld_sketch * num_cols_sketch);
        memory::malloc(&dresult_rp, ld_r_factor * num_cols_mtx);
        memory::malloc_cpu(&tau, num_rows_sketch);
    }

    void free()
    {
        memory::free(dmtx_rp);
        memory::free(dsketch_rp);
        memory::free(dresult_rp);
        memory::free_cpu(tau);
    }
};

}  // namespace generalized_split

template <typename value_type_in, typename value_type,
          typename index_type, ContextType device_type=CUDA>
void compute_precond(index_type num_rows_sketch, index_type num_cols_sketch,
                  value_type* dsketch, index_type ld_sketch,
                  index_type num_rows_mtx, index_type num_cols_mtx,
                  value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
                  index_type ld_r_factor,
                  generalized_split::state<value_type_in, value_type,
                                           index_type>* precond_state,
                  std::shared_ptr<Context<device_type>> context, double* runtime,
                  double* t_mm, double* t_qr);


template <typename value_type_in, typename value_type, typename index_type, ContextType device_type=CUDA>
class GeneralizedSplit
    : public preconditioner<value_type, index_type, device_type> {
public:

    void generate() {
        auto context = this->context_;
        auto num_rows = this->mtx_->get_size()[0];
        auto num_cols = this->mtx_->get_size()[1];
        index_type sampled_rows = (index_type)(sampling_coeff_ * num_cols);
        auto size = this->mtx_->get_size();
        auto queue = context->get_queue();
        std::shared_ptr<Context<CPU>> context_cpu = Context<CPU>::create();

        // Generates sketch matrix.
        sketch_mtx_ = matrix::Dense<value_type, device_type>::create(context, {sampled_rows, num_rows});
        if (std::is_same<value_type, double>::value) {
            auto status  = curandGenerateNormalDouble(context->get_generator(),
                                       (double*)sketch_mtx_->get_values(),
                                       sampled_rows * num_rows, 0, 1);
        } else if (std::is_same<value_type, float>::value) {
            curandGenerateNormal(context->get_generator(),
                                 (float*)sketch_mtx_->get_values(),
                                 sampled_rows * num_rows, 0, 1);
        }
        cudaDeviceSynchronize();

        // Allocates memory for matrices.
        this->precond_mtx_ = matrix::Dense<value_type, device_type>::create(context, {sampled_rows, size[1]});
        this->precond_mtx_internal_ = matrix::Dense<value_type_in, device_type>::create(context, {sampled_rows, size[1]});
        this->dt_ = matrix::Dense<value_type, device_type>::create(context, size);
        this->mtx_rp_ = matrix::Dense<value_type_in, device_type>::create(context, size);
        this->dsketch_rp_ = matrix::Dense<value_type_in, device_type>::create(context, {sampled_rows, size[0]});
        this->dresult_rp_ = matrix::Dense<value_type_in, device_type>::create(context, {sampled_rows, size[1]});
        dim2 s = {size[0], 1};
        this->tau_ = matrix::Dense<value_type, CPU>::create(context_cpu, s);

        // experimental
        this->temp_ = matrix::Dense<value_type, device_type>::create(context, {size[1], 1});
        this->temp_mtx_ = matrix::Dense<value_type, device_type>::create(context, {size[1], size[1]});
        this->t = matrix::Dense<value_type, device_type>::create(this->context_,
                    {size[1], 1});
    }

    static std::unique_ptr<
        GeneralizedSplit<value_type_in, value_type, index_type>>
    create(std::shared_ptr<Context<device_type>> context)
    {
        return std::unique_ptr<
            GeneralizedSplit<value_type_in, value_type, index_type>>(
            new GeneralizedSplit<value_type_in, value_type, index_type>(
                context));
    }

    static std::unique_ptr<
        GeneralizedSplit<value_type_in, value_type, index_type>>
    create(std::shared_ptr<matrix::Dense<value_type, device_type>> mtx)
    {
        return std::unique_ptr<
            GeneralizedSplit<value_type_in, value_type, index_type>>(
            new GeneralizedSplit<value_type_in, value_type, index_type>(mtx));
    }

    static std::unique_ptr<
        GeneralizedSplit<value_type_in, value_type, index_type>>
    create(std::shared_ptr<matrix::Dense<value_type, device_type>> mtx,
        double sampling_coeff)
    {
        return std::unique_ptr<
            GeneralizedSplit<value_type_in, value_type, index_type>>(
            new GeneralizedSplit<value_type_in, value_type, index_type>(mtx,
                sampling_coeff));
    }

    static std::unique_ptr<
        GeneralizedSplit<value_type_in, value_type, index_type>>
    create(std::shared_ptr<Context<device_type>> context,
           std::shared_ptr<matrix::Dense<value_type, device_type>> mtx)
    {
        return std::unique_ptr<
            GeneralizedSplit<value_type_in, value_type, index_type>>(
            new GeneralizedSplit<value_type_in, value_type, index_type>(mtx));
    }

    void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u)
    {
        auto queue = this->context_->get_queue();
        auto size = this->precond_mtx_->get_size();
        auto first_index = this->mtx_->get_size()[0];
        auto u1 = &u_vector[first_index];

        // option-1:
        // blas::trsv(MagmaUpper, trans, MagmaNonUnit, size[1],
            //  this->precond_mtx_->get_values(), size[0], u1, inc_u, queue);

        // option-2:
        // blas::trmv(MagmaUpper, trans, MagmaNonUnit,
            // size[1], this->precond_mtx_->get_values(), size[0], u1, 1, queue);

        // option-3:
        {
            auto t = matrix::Dense<value_type, device_type>::create(this->context_,
                {size[1], 1});
            blas::copy(size[1], u1, 1, t->get_values(), 1, queue);
            blas::gemv(trans, size[0], size[1], 1.0, this->precond_mtx_->get_values(),
                       size[0], t->get_values(), 1, 0.0, u1, 1, queue);
        }
    }

    void compute() {
        if (this->context_->is_tf32_used() == true) {
            this->context_->use_tf32_math_operations();
        }
        value_type* hat_mtx = nullptr;
        double runtime;
        auto precond_state = new state<value_type_in, value_type,
            index_type>();
        double runtime_local;
        double t_mm;
        double t_qr;
        precond_state->allocate(this->mtx_->get_size()[0], this->mtx_->get_size()[1],
            sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1],
            sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[0]);

        for (auto i = 0; i < this->logger_.warmup_runs_; i++) {
            compute_precond(sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1],
                sketch_mtx_->get_values(), sketch_mtx_->get_size()[0],
                this->mtx_->get_size()[0], this->mtx_->get_size()[1],
                this->mtx_->get_values(), this->mtx_->get_size()[0],
                this->precond_mtx_->get_values(), this->precond_mtx_->get_size()[0],
                precond_state,
                this->context_, &this->logger_.runtime_, &this->logger_.runtime_sketch_, &this->logger_.runtime_qr_);
        }
        this->logger_.runtime_ = 0.0;
        this->logger_.runtime_sketch_ = 0.0;
        this->logger_.runtime_qr_ = 0.0;

        for (auto i = 0; i < this->logger_.runs_; i++) {
            compute_precond(sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1],
                sketch_mtx_->get_values(), sketch_mtx_->get_size()[0],
                this->mtx_->get_size()[0], this->mtx_->get_size()[1],
                this->mtx_->get_values(), this->mtx_->get_size()[0],
                this->precond_mtx_->get_values(), this->precond_mtx_->get_size()[0],
                precond_state,
                this->context_, &this->logger_.runtime_, &this->logger_.runtime_sketch_, &this->logger_.runtime_qr_);
        }
        this->logger_.runtime_ = this->logger_.runtime_ / this->logger_.runs_;
        this->logger_.runtime_sketch_ = this->logger_.runtime_sketch_ / this->logger_.runs_;
        this->logger_.runtime_qr_ = this->logger_.runtime_qr_ / this->logger_.runs_;

        precond_state->free();

        magma_int_t status;
        dim2 s = {this->precond_mtx_->get_size()[1], this->precond_mtx_->get_size()[1]};
        cuda::set_upper_triang(s, this->precond_mtx_->get_values(), this->precond_mtx_->get_size()[0]);
        blas::trtri(MagmaUpper, MagmaNonUnit,
                    this->precond_mtx_->get_size()[1],
                    this->precond_mtx_->get_values(),
                    this->precond_mtx_->get_size()[0],
                    &status);
        if (this->context_->is_tf32_used() == true) {
            this->context_->disable_tf32_math_operations();
        }
    }

    template<typename value_type_out>
    std::shared_ptr<GeneralizedSplit<value_type_out, value_type_out, index_type, device_type>> convert_to() {
        std::shared_ptr<GeneralizedSplit<value_type_out, value_type_out, index_type, device_type>> tmp = GeneralizedSplit<value_type_in, value_type_out, index_type, device_type>::create();
        tmp->mtx_->copy_from(this->mtx_);
        tmp->precond_mtx_->copy_from(this->precond_mtx_);
        return tmp;
    }

    void set_logger(logger& logger) {
        this->logger_ = logger;
    }

    matrix::Dense<value_type, device_type>* get_mtx() { return this->precond_mtx_.get(); }

    value_type* get_values() { return this->precond_mtx_->get_values(); }

    dim2 get_size() { return this->precond_mtx_->get_size(); }

private:

    GeneralizedSplit(std::shared_ptr<Context<device_type>> context)
    {
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = 1.5;
        this->context_ = context;
    }

    GeneralizedSplit(std::shared_ptr<Context<device_type>> context,
        std::shared_ptr<matrix::Dense<value_type, device_type>> mtx)
    {
        this->context_ = context;
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = 1.5;
        this->mtx_ = mtx;
    }

    GeneralizedSplit(std::shared_ptr<matrix::Dense<value_type, device_type>> mtx)
    {
        this->context_ = mtx->get_context();
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = 1.5;
        this->mtx_ = mtx;
    }

    GeneralizedSplit(std::shared_ptr<matrix::Dense<value_type, device_type>> mtx,
        double sampling_coeff)
    {
        this->context_ = mtx->get_context();
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = sampling_coeff;
        this->mtx_ = mtx;
    }

    GeneralizedSplit(matrix::Dense<value_type, device_type>& mtx, matrix::Dense<value_type, device_type>& sketch_mtx)
    {
        this->context_ = mtx->get_context();
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = 1.5;
        this->mtx_ = mtx;
        this->sketch_mtx_ = sketch_mtx;
    }

    value_type sampling_coeff_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> sketch_mtx_;
    std::unique_ptr<matrix::Dense<value_type, device_type>> dt_;
    std::unique_ptr<matrix::Dense<value_type, device_type>> temp_;
    std::unique_ptr<matrix::Dense<value_type, device_type>> temp_mtx_;
    std::unique_ptr<matrix::Dense<value_type, device_type>> t;
    std::unique_ptr<matrix::Dense<value_type_in, device_type>> mtx_rp_;
    std::unique_ptr<matrix::Dense<value_type_in, device_type>> dsketch_rp_;
    std::unique_ptr<matrix::Dense<value_type_in, device_type>> dresult_rp_;
    std::unique_ptr<matrix::Dense<value_type_in, device_type>> precond_mtx_internal_;
    std::unique_ptr<matrix::Dense<value_type, CPU>> tau_;
};


}  // namespace preconditioner
}  // namespace rls

#endif
