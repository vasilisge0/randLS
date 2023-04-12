#ifndef RLS_GENERALIZED_SPLIT_HPP
#define RLS_GENERALIZED_SPLIT_HPP


#include <iostream>
#include <memory>
#include "magma_v2.h"


#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../utils/io.hpp"
#include "../blas/blas.hpp"
#include "../dense/dense.hpp"
#include "../memory/magma_context.hpp"
#include "preconditioner.hpp"


namespace rls {
namespace preconditioner {
namespace generalized_split {

template <typename value_type_internal, typename value_type,
          typename index_type>
struct state {
    value_type_internal* dmtx_rp = nullptr;
    value_type_internal* dsketch_rp = nullptr;
    value_type_internal* dresult_rp = nullptr;
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

template <typename value_type_internal, typename value_type,
          typename index_type, ContextType device_type=CUDA>
void generate_old(index_type num_rows_sketch, index_type num_cols_sketch,
                  value_type* dsketch, index_type ld_sketch,
                  index_type num_rows_mtx, index_type num_cols_mtx,
                  value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
                  index_type ld_r_factor,
                  generalized_split::state<value_type_internal, value_type,
                                           index_type>* precond_state,
                  std::shared_ptr<Context<device_type>> context, double* runtime,
                  double* t_mm, double* t_qr);

template <typename value_type_internal, typename value_type,
          typename index_type, ContextType device_type=CUDA>
void generate_old(index_type num_rows_sketch, index_type num_cols_sketch,
                  value_type* dsketch, index_type ld_sketch,
                  index_type num_rows_mtx, index_type num_cols_mtx,
                  value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
                  index_type ld_r_factor, value_type* hat_mtx,
                  std::shared_ptr<Context<device_type>> context);

// template <typename value_type_in, typename value_type,
//   typename index_type>
// void compute_precond(index_type num_rows_sketch, index_type num_cols_sketch,
// value_type* dsketch, index_type ld_sketch,
//   index_type num_rows_mtx, index_type num_cols_mtx, value_type* dmtx,
//   index_type ld_mtx, value_type* dr_factor, index_type ld_r_factor,
//   value_type* hat_mtx, detail::magma_info& info);

template <typename value_type_in, typename value_type, typename index_type, ContextType device_type=CUDA>
void compute_precond(index_type num_rows_sketch, index_type num_cols_sketch,
                     value_type* dsketch, index_type ld_sketch,
                     index_type num_rows_mtx, index_type num_cols_mtx,
                     value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
                     index_type ld_r_factor, value_type* hat_mtx, Context<device_type>& info,
                     double* runtime);


template <typename value_type_in, typename value_type, typename index_type, ContextType device_type=CUDA>
class GeneralizedSplit
    : public preconditioner<value_type_in, value_type, index_type, device_type> {
public:
    GeneralizedSplit()
    {
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = 1.5;
    }

    GeneralizedSplit(std::shared_ptr<matrix::Dense<value_type, device_type>> mtx)
    {
        this->context_ = mtx->get_context();
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = 1.5;
        this->mtx_ = mtx;
    }

    GeneralizedSplit(std::shared_ptr<Context<device_type>> context)
    {
        this->precond_type_ = GENERALIZED_SPLIT;
        this->sampling_coeff_ = 1.5;
        this->context_ = context;
    }

    GeneralizedSplit(matrix::Dense<value_type, device_type>& sketch_mtx)
    {
        sketch_mtx_ = std::move(sketch_mtx);
    }

    GeneralizedSplit(std::shared_ptr<matrix::Dense<value_type, device_type>> mtx,
                     std::shared_ptr<matrix::Dense<value_type, device_type>> sketch_mtx)
    {
        // this->mtx_ = mtx;
        // this->sketch_mtx_ = sketch_mtx;
        // auto size = this->mtx_->get_size();
// 
        // auto sampled_rows = sketch_mtx_->get_size()[0];
        // this->precond_mtx_ = std::shared_ptr<matrix::Dense<value_type, device_type>>(
            // new matrix::Dense<value_type, device_type>());
        // this->precond_mtx_internal_ =
            // std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
                // new matrix::Dense<value_type_in, device_type>());
        // dt_ = std::unique_ptr<matrix::Dense<value_type, device_type>>(
            // new matrix::Dense<value_type, device_type>());
        // mtx_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
            // new matrix::Dense<value_type_in, device_type>());
        // dsketch_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
            // new matrix::Dense<value_type_in, device_type>());
        // dresult_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
            // new matrix::Dense<value_type_in, device_type>());
        // tau_ = std::unique_ptr<matrix::Dense<index_type>>(
            // new matrix::Dense<value_type>());
// 
        // this->precond_mtx_->generate({sampled_rows, size[1]});
        // this->precond_mtx_internal_->generate({sampled_rows, size[1]});
        // dt_->generate(size);
        // mtx_rp_->generate(size);
        // dsketch_rp_->generate({sampled_rows, size[0]});
        // dresult_rp_->generate({sampled_rows, size[1]});
        // tau_->generate_cpu({size[0], 1});
    }

    ~GeneralizedSplit() {}

    void generate(double* runtime);

    void generate(matrix::Dense<value_type, device_type>* mtx)
    {
        // auto context = this->context_;
        // auto num_rows = mtx->get_size()[0];
        // auto num_cols = mtx->get_size()[1];
        // index_type sampled_rows = (index_type)(sampling_coeff_ * num_cols);
        // auto size = mtx->get_size();

        // // Generates sketch matrix.
        // sketch_mtx_ = std::shared_ptr<rls::matrix::Dense<value_type, device_type>>(
        //     new rls::matrix::Dense<value_type, device_type>({sampled_rows, num_rows}));
        // sketch_mtx_->generate();
        // if (std::is_same<value_type, double>::value) {
        //     curandGenerateNormalDouble(context->get_generator(),
        //                                (double*)sketch_mtx_->get_values(),
        //                                sampled_rows * num_rows, 0, 1);
        // } else if (std::is_same<value_type, float>::value) {
        //     curandGenerateNormal(context->get_generator(),
        //                          (float*)sketch_mtx_->get_values(),
        //                          sampled_rows * num_rows, 0, 1);
        // }
        // cudaDeviceSynchronize();
        // // Construct preconditioner
        // this->precond_mtx_ = std::shared_ptr<matrix::Dense<value_type, device_type>>(
        //     new matrix::Dense<value_type, device_type>({sampled_rows, size[1]}));
        // this->precond_mtx_internal_ =
        //     std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
        //         new matrix::Dense<value_type_in, device_type>({sampled_rows, size[1]}));
        // dt_ = std::unique_ptr<matrix::Dense<value_type, device_type>>(
        //     new matrix::Dense<value_type, device_type>(size));
        // mtx_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
        //     new matrix::Dense<value_type_in, device_type>(size));
        // dsketch_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
        //     new matrix::Dense<value_type_in, device_type>({sampled_rows, size[0]}));
        // dresult_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
        //     new matrix::Dense<value_type_in, device_type>({sampled_rows, size[1]}));
        // tau_ = std::unique_ptr<matrix::Dense<index_type>>(
        //     new matrix::Dense<value_type>({size[0], 1}));

        // // Generate matrices.
        // this->precond_mtx_->generate();
        // this->precond_mtx_internal_->generate();
        // dt_->generate();
        // mtx_rp_->generate();
        // dsketch_rp_->generate();
        // dresult_rp_->generate();
        // tau_->generate_cpu();
    }

    GeneralizedSplit<value_type_in, value_type, index_type>* get()
    {
        return this;
    }

    matrix::Dense<value_type, device_type>* get_mtx() { return this->precond_mtx_.get(); }

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

    void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u)
    {
        auto queue = this->context_->get_queue();
        auto size = this->precond_mtx_->get_size();
        auto first_index = this->mtx_->get_size()[0];
        auto u1 = &u_vector[first_index];
        blas::trsv(MagmaUpper, trans, MagmaNonUnit, size[1],
             this->precond_mtx_->get_values(), size[0], u1, inc_u, queue);
    }


    void compute(matrix::Dense<value_type, device_type>* dmtx)
    {
        value_type* hat_mtx = nullptr;
        double runtime;
        auto precond_state = new state<value_type_in, value_type,
        index_type>(); double runtime_local; double t_mm; double t_qr;
        precond_state->allocate(dmtx->get_size()[0], dmtx->get_size()[1],
            sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1],
            sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[0]);
        generate_old(sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1],
            sketch_mtx_->get_values(), sketch_mtx_->get_size()[0],
            dmtx->get_size()[0], dmtx->get_size()[1],
            dmtx->get_values(), dmtx->get_size()[0],
        this->precond_mtx_->get_values(), this->precond_mtx_->get_size()[0],
            precond_state,
        this->context_, &runtime_local, &t_mm, &t_qr);
            precond_state->free();
    }

    void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u,
               magma_queue_t queue)
    {}

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

        // Construct preconditioner
        this->precond_mtx_ = matrix::Dense<value_type, device_type>::create(context, {sampled_rows, size[1]});
        this->precond_mtx_internal_ = matrix::Dense<value_type, device_type>::create(context, {sampled_rows, size[1]});
        this->dt_ = matrix::Dense<value_type, device_type>::create(context, size);
        this->mtx_rp_ = matrix::Dense<value_type, device_type>::create(context, size);
        this->dsketch_rp_ = matrix::Dense<value_type, device_type>::create(context, {sampled_rows, size[0]});
        this->dresult_rp_ = matrix::Dense<value_type, device_type>::create(context, {sampled_rows, size[1]});
        dim2 s = {size[0], 1};
        this->tau_ = matrix::Dense<value_type, CPU>::create(context_cpu, s);

        // this->precond_mtx_ = std::shared_ptr<matrix::Dense<value_type, device_type>>(
        //    new matrix::Dense<value_type, device_type>(context, {sampled_rows, size[1]}));
        // this->precond_mtx_internal_ =
        //    std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
            //    new matrix::Dense<value_type_in, device_type>(context, {sampled_rows, size[1]}));
        // dt_ = std::unique_ptr<matrix::Dense<value_type, device_type>>(
        //    new matrix::Dense<value_type, device_type>(context, size));
        // mtx_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
        //    new matrix::Dense<value_type_in, device_type>(context, size));
        // dsketch_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
        //    new matrix::Dense<value_type_in, device_type>(context, {sampled_rows, size[0]}));
        // dresult_rp_ = std::unique_ptr<matrix::Dense<value_type_in, device_type>>(
        //    new matrix::Dense<value_type_in, device_type>(context, {sampled_rows, size[1]}));
        // tau_ = std::unique_ptr<matrix::Dense<index_type>>(
        //    new matrix::Dense<index_type>(context, {size[0], 1}));
        // Generate matrices.
        //this->precond_mtx_->generate();
        //this->precond_mtx_internal_->generate();
        //dt_->generate();
        //mtx_rp_->generate();
        //dsketch_rp_->generate();
        //dresult_rp_->generate();
        //tau_->generate_cpu();
    }

    void compute() {
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
        generate_old(sketch_mtx_->get_size()[0], sketch_mtx_->get_size()[1],
            sketch_mtx_->get_values(), sketch_mtx_->get_size()[0],
            this->mtx_->get_size()[0], this->mtx_->get_size()[1],
            this->mtx_->get_values(), this->mtx_->get_size()[0],
            this->precond_mtx_->get_values(), this->precond_mtx_->get_size()[0],
            precond_state,
            this->context_, &runtime_local, &t_mm, &t_qr);
        precond_state->free();
    }

    value_type* get_values() { return this->precond_mtx_->get_values(); }

    dim2 get_size() { return this->precond_mtx_->get_size(); }

private:
    value_type sampling_coeff_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> sketch_mtx_;
    std::unique_ptr<matrix::Dense<value_type, device_type>> dt_;
    std::unique_ptr<matrix::Dense<value_type_in, device_type>> mtx_rp_;
    std::unique_ptr<matrix::Dense<value_type_in, device_type>> dsketch_rp_;
    std::unique_ptr<matrix::Dense<value_type_in, device_type>> dresult_rp_;
    std::unique_ptr<matrix::Dense<value_type, CPU>> tau_;
};


}  // namespace preconditioner
}  // namespace rls

#endif
