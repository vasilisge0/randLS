#ifndef RLS_SKETCHQR_HPP
#define RLS_SKETCHQR_HPP


#include <iostream>
#include <memory>
#include "magma_v2.h"


#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../utils/io.hpp"
#include "../blas/blas.hpp"
#include "../matrix/dense/dense.hpp"
#include "../memory/magma_context.hpp"
#include "../sketch/gaussian.hpp"
#include "preconditioner.hpp"


namespace rls {
namespace preconditioner {


template <typename vtype>
void apply_impl(vtype t);

template <typename vtype, typename vtype_internal,
          typename index_type>
class SketchQrLogger : public preconditioner::logger {};

template <typename vtype_0, typename vtype_in_0,
          typename vtype_precond_apply, typename index_type>
class SketchQrConfig : public Config {

private:
    SketchQrConfig(double sampling_coefficient) : Config(sampling_coefficient) {}

public:
    static std::unique_ptr<SketchQrConfig<vtype_0, vtype_in_0, vtype_precond_apply, index_type>>
        create(double sampling_coefficient)
    {
        return std::unique_ptr<SketchQrConfig<vtype_0, vtype_in_0, vtype_precond_apply, index_type>>
            (new SketchQrConfig<vtype_0, vtype_in_0, vtype_precond_apply, index_type>(sampling_coefficient));
    }
};


template <ContextType device, typename vtype,
          typename vtype_internal, typename index_type>
struct state {
    size_t num_elems_internal = 0;
    size_t num_elems   = 0;
    size_t ld_mtx      = 0;
    size_t ld_sketch   = 0;
    size_t rows_sketch = 0;
    size_t ld_r_factor = 0;
    dim2 size          = {0, 0};
    std::shared_ptr<matrix::Dense<CPU, vtype>> tau;
    std::shared_ptr<matrix::Dense<device, vtype_internal>>
        sketched_mtx;                   // That is always dense.
    std::shared_ptr<MtxOp<device>> mtx; // That could be sparse though.

    state(MtxOp<device>* mtx_in, dim2 size, size_t rows_sketch, size_t ld_sketch, size_t ld_mtx)
    {
        this->rows_sketch = rows_sketch;
        this->ld_mtx      = ld_mtx;
        this->size        = size;
        this->ld_r_factor = ld_sketch;
        auto context = mtx_in->get_context();
        // Here this could be dense or sparse.
        if (auto t_in = dynamic_cast<matrix::Dense<device, vtype>*>(mtx_in); t_in != nullptr) {
            mtx = matrix::Dense<device, vtype_internal>::create(
                context, {this->size[0], this->size[1]});
            auto t = static_cast<matrix::Dense<device, vtype_internal>*>(mtx.get());
            static_cast<matrix::Dense<device, vtype_internal>*>(mtx.get())->copy_from(t_in);
        }
        else if (auto t_in = dynamic_cast<matrix::Sparse<device, vtype, index_type>*>(mtx_in); t_in != nullptr) {
            mtx = matrix::Sparse<device, vtype_internal, index_type>::create(
                context, {this->size[0], this->size[1]}, mtx_in->get_num_elems());
            static_cast<matrix::Sparse<device, vtype_internal, index_type>*>(mtx.get())->copy_from(t_in);
            auto queue = context->get_queue();
        }
        sketched_mtx = matrix::Dense<device, vtype_internal>::create(
            context, {static_cast<int>(this->ld_r_factor), this->size[1]});
        auto context_ref = rls::share(rls::Context<CPU>::create());
        tau = matrix::Dense<rls::CPU, vtype>::create(context_ref, {static_cast<int>(rows_sketch), 1});
    }
};  // end of struct state


template <ContextType device, typename vtype,
          typename vtype_internal, typename index_type>
int sketch_qr_impl(
    std::shared_ptr<Context<device>> context,
    std::shared_ptr<
        SketchOperator<device, vtype_internal, index_type>>
        sketch,
    std::shared_ptr<MtxOp<device>> mtx,
    std::shared_ptr<matrix::Dense<device, vtype>> precond_mtx,
    state<device, vtype, vtype_internal, index_type>* state);


template <ContextType device, typename vtype,
          typename vtype_in, typename vtype_precond_apply, typename index_type>
class SketchQr : public PrecondOperator<device, vtype_precond_apply, index_type> {
public:
    static std::unique_ptr<SketchQr<device, vtype, vtype_in, vtype_precond_apply, index_type>> create( std::shared_ptr<MtxOp<device>> mtx,
            std::shared_ptr<SketchOperator<device, vtype_in, index_type>>
                sketch,
            std::shared_ptr<Config> config);

    static std::unique_ptr<SketchQr<device, vtype, vtype_in, vtype_precond_apply, index_type>> create(
        std::shared_ptr<MtxOp<device>> mtx,
        std::string& filename_precond,
        std::shared_ptr<Config> config);

    void generate();

    void apply(std::shared_ptr<Context<device>> context, magma_trans_t trans,
        rls::matrix::Dense<device, vtype_precond_apply>* u_vector_part_1);

    matrix::Dense<device, vtype>* get_precond_mtx()
    {
        return precond_mtx_.get();
    }

    ~SketchQr();

private:

    SketchQr(
        std::shared_ptr<MtxOp<device>> mtx,
        std::shared_ptr<SketchOperator<device, vtype_in, index_type>>
            sketch,
        std::shared_ptr<SketchQrConfig<vtype, vtype_in, vtype_precond_apply, index_type>>
            config);

    SketchQr(
        std::shared_ptr<MtxOp<device>> mtx,
        std::string& filename_precond,
        std::shared_ptr<SketchQrConfig<vtype, vtype_in, vtype_precond_apply, index_type>>
            config);

    std::shared_ptr<Context<device>> context_;
    std::shared_ptr<SketchQrConfig<vtype, vtype_in, vtype_precond_apply, index_type>>
        config_;
    SketchQrLogger<vtype, vtype_in, index_type> logger_;
    state<device, vtype, vtype_in, index_type>* state_;
    std::shared_ptr<SketchOperator<device, vtype_in, index_type>> sketch_;
    std::shared_ptr<MtxOp<device>> mtx_;
    std::shared_ptr<matrix::Dense<device, vtype>> precond_mtx_;
    std::shared_ptr<matrix::Dense<device, vtype_precond_apply>>
        precond_mtx_apply_;
    std::shared_ptr<matrix::Dense<device, vtype_precond_apply>> t_apply;
    std::shared_ptr<matrix::Dense<device, vtype>> t;
    std::shared_ptr<matrix::Dense<device, vtype>> t0;
};

//template<> class SketchQr<CUDA, double, double, double, magma_int_t>;


}  // end of namespace preconditioner
}  // end of namespace rls


#endif
