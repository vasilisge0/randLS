#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../cuda/solver/lsqr_kernels.cuh"
#include "../../include/base_types.hpp"
#include "../../utils/convert.hpp"
#include "../../utils/io.hpp"
#include "../sketch/gaussian.hpp"
#include "../blas/blas.hpp"
#include "../memory/magma_context.hpp"
#include "sketchqr.hpp"


namespace rls {
namespace preconditioner {


template <ContextType device, typename vtype,
         typename vtype_internalternal, typename index_type>
int sketch_qr_impl(
    std::shared_ptr<Context<device>> context,
    std::shared_ptr<
        SketchOperator<device, vtype_internalternal, index_type>>
        sketch,
    std::shared_ptr<MtxOp<device>> mtx,
    std::shared_ptr<matrix::Dense<device, vtype>> precond_mtx,
    state<device, vtype, vtype_internalternal, index_type>* state)
{
    auto num_rows_mtx = mtx->get_size()[0];
    auto num_rows_sketch = state->sketched_mtx->get_size()[0];
    auto num_cols_mtx = state->sketched_mtx->get_size()[1];
    auto ld_r_factor = state->sketched_mtx->get_ld();
    std::cout << "before apply\n";
    sketch->apply(state->mtx, state->sketched_mtx); // Copy to state->mtx outside of sketch_qr_impl.
    std::cout << "after apply\n";
    index_type info_qr = 0;
    precond_mtx->copy_from(state->sketched_mtx.get());
    blas::geqrf2(context, num_rows_sketch, num_cols_mtx,
                 precond_mtx->get_values(), ld_r_factor,
                 state->tau->get_values(), &info_qr);
    dim2 s = {ld_r_factor, num_cols_mtx};
    cuda::set_upper_triang(s, precond_mtx->get_values(), precond_mtx->get_size()[0]);
    auto queue = context->get_queue();
    std::cout << "\n\n\nprecond:\n";
    io::print_mtx_gpu(3, 3, precond_mtx->get_values(), precond_mtx->get_ld(), queue);
    std::cout << "\n\n\n\n";
    return info_qr;
}

template int sketch_qr_impl(
    std::shared_ptr<Context<CUDA>> context,
    std::shared_ptr<SketchOperator<CUDA, double, magma_int_t>> sketch,
    std::shared_ptr<MtxOp<CUDA>> mtx,
    std::shared_ptr<matrix::Dense<CUDA, double>> precond_mtx,
    state<CUDA, double, double, magma_int_t>* state);

template int sketch_qr_impl(
    std::shared_ptr<Context<CUDA>> context,
    std::shared_ptr<SketchOperator<CUDA, float, magma_int_t>> sketch,
    std::shared_ptr<MtxOp<CUDA>> mtx,
    std::shared_ptr<matrix::Dense<CUDA, double>> precond_mtx,
    state<CUDA, double, float, magma_int_t>* state);

template int sketch_qr_impl(
    std::shared_ptr<Context<CUDA>> context,
    std::shared_ptr<SketchOperator<CUDA, __half, magma_int_t>> sketch,
    std::shared_ptr<MtxOp<CUDA>> mtx,
    std::shared_ptr<matrix::Dense<CUDA, double>> precond_mtx,
    state<CUDA, double, __half, magma_int_t>* state);

template int sketch_qr_impl(
    std::shared_ptr<Context<rls::CUDA>> context,
    std::shared_ptr<
        SketchOperator<rls::CUDA, double, magma_int_t>>
        sketch,
    std::shared_ptr<MtxOp<rls::CUDA>> mtx,
    std::shared_ptr<matrix::Dense<rls::CUDA, float>> precond_mtx,
    state<rls::CUDA, float, double, magma_int_t>* state);


template<ContextType device,
         typename vtype,
         typename vtype_internal,
         typename vtype_precond_apply,
         typename index_type>
std::unique_ptr<SketchQr<device, vtype, vtype_internal,
                                vtype_precond_apply, index_type>> SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>::create(
        std::shared_ptr<MtxOp<device>> mtx,
        std::shared_ptr<SketchOperator<device, vtype_internal, index_type>>
            sketch,
        std::shared_ptr<Config> config)
{
    auto t = std::static_pointer_cast<SketchQrConfig<vtype, vtype_internal, vtype_precond_apply, index_type>>(config);
    return std::unique_ptr<SketchQr<device, vtype, vtype_internal,
                                    vtype_precond_apply, index_type>>(
        new SketchQr<device, vtype, vtype_internal,
                     vtype_precond_apply, index_type>(mtx, sketch, t));
}

template<ContextType device,
         typename vtype,
         typename vtype_internal,
         typename vtype_precond_apply,
         typename index_type>
static std::unique_ptr<SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>> 
    SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>::create(
    std::shared_ptr<MtxOp<device>> mtx,
    std::string& filename_precond,
    std::shared_ptr<Config> config)
{
    auto t = std::static_pointer_cast<SketchQrConfig<vtype, vtype_internal, vtype_precond_apply, index_type>>(config);
    return std::unique_ptr<SketchQr<device, vtype, vtype_internal,
                                    vtype_precond_apply, index_type>>(
        new SketchQr<device, vtype, vtype_internal,
                     vtype_precond_apply, index_type>(mtx, filename_precond, t));
}

template <ContextType device, typename vtype,
      typename vtype_internal, typename vtype_precond_apply,
      typename index_type>
void SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>::generate()
{
    //@error here
    std::cout << "before sketch_qr_impl\n";
    auto info_qr = sketch_qr_impl(context_, sketch_, mtx_, precond_mtx_, state_);
    std::cout << "after sketch_qr_impl\n";
    if (info_qr != 0) {
        magma_xerbla("geqrf2_gpu", info_qr);
    }
    //@error here
    blas::trtri(context_, MagmaUpper, MagmaNonUnit, precond_mtx_->get_size()[1],
                precond_mtx_->get_values(), precond_mtx_->get_ld(), &info_qr);
    auto queue = context_->get_queue();
    //io::write_mtx("precond_hgdp.mtx", precond_mtx_->get_size()[0], precond_mtx_->get_size()[1],
    //    (double*)precond_mtx_->get_values(), precond_mtx_->get_ld(), queue);
    precond_mtx_apply_->copy_from(precond_mtx_.get());
}


template <ContextType device, typename vtype,
      typename vtype_internal, typename vtype_precond_apply,
      typename index_type>
void SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>::apply(std::shared_ptr<Context<device>> context, magma_trans_t trans,
        rls::matrix::Dense<device, vtype_precond_apply>* u_vector_part_1)
{
    auto exec = context->get_executor();
    auto size = precond_mtx_->get_size();
    vtype_precond_apply one = 1.0;
    vtype_precond_apply zero = 0.0;
    auto queue = context->get_queue();
    //io::print_mtx_gpu(size[1], 1, precond_mtx_apply_->get_values() + size[0]*size[1] - size[0], size[0], queue);
    //
    blas::gemv(context, trans, size[1], size[1], one, precond_mtx_apply_->get_values(),
        size[0], u_vector_part_1->get_values(), 1, zero, t_apply->get_values(), 1);
    cudaDeviceSynchronize();
    exec->copy(size[1], t_apply->get_values(), u_vector_part_1->get_values());
}

template <ContextType device, typename vtype,
      typename vtype_internal, typename vtype_precond_apply,
      typename index_type>
SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>::~SketchQr() {
}

template <ContextType device, typename vtype,
      typename vtype_internal, typename vtype_precond_apply,
      typename index_type>
SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>::SketchQr(
        std::shared_ptr<MtxOp<device>> mtx,
        std::shared_ptr<SketchOperator<device, vtype_internal, index_type>>
            sketch,
        std::shared_ptr<SketchQrConfig<vtype, vtype_internal, vtype_precond_apply, index_type>>
            config) : PrecondOperator<device, vtype_precond_apply, index_type>(mtx)
{
std::cout << "in sketchqr constructor\n";
    context_ = mtx->get_context();
    mtx_ = mtx;
    config_ = config;
    auto size = this->mtx_->get_size();
    auto sampled_rows =
        std::ceil(config_->get_sampling_coefficient() * size[1]);
    state_ = new state<device, vtype, vtype_internal, index_type>(
        mtx_.get(), size, sampled_rows, sampled_rows, mtx->get_size()[0]);
    t_apply = matrix::Dense<device, vtype_precond_apply>::create(context_, dim2(size[1], 1));
    sketch_ = sketch;
    this->precond_mtx_ = matrix::Dense<device, vtype>::create(context_, dim2(sketch->get_size()[0], mtx->get_size()[1]));
    this->precond_mtx_apply_ = matrix::Dense<device, vtype_precond_apply>::create(context_, {sketch->get_size()[0], mtx->get_size()[1]});
    std::cout << "before generate\n";
    generate();
    std::cout << "after generate\n";
    auto queue = context_->get_queue();
    io::print_mtx_gpu(6, 6, precond_mtx_->get_values(), precond_mtx_->get_ld(), queue);
}

template <ContextType device, typename vtype,
      typename vtype_internal, typename vtype_precond_apply,
      typename index_type>
SketchQr<device, vtype, vtype_internal, vtype_precond_apply, index_type>::SketchQr(
        std::shared_ptr<MtxOp<device>> mtx,
        std::string& filename_precond,
        std::shared_ptr<SketchQrConfig<vtype, vtype_internal, vtype_precond_apply, index_type>>
            config) : PrecondOperator<device, vtype_precond_apply, index_type>(mtx)
{
    context_ = mtx->get_context();
    mtx_ = mtx;
    config_ = config;
    auto size = this->mtx_->get_size();
    t_apply = matrix::Dense<device, vtype_precond_apply>::create(context_, dim2(size[1], 1));
    auto sampled_rows = t_apply->get_size()[0];
    state_ = new state<device, vtype, vtype_internal, index_type>(
        mtx_.get(), size, sampled_rows, sampled_rows, mtx->get_size()[0]);
    this->precond_mtx_ = matrix::Dense<device, vtype>::create(context_, filename_precond);
    this->precond_mtx_apply_ = matrix::Dense<device, vtype_precond_apply>::create(context_, {precond_mtx_->get_size()[0], precond_mtx_->get_size()[1]});
    this->precond_mtx_apply_->copy_from(precond_mtx_.get());
}

template class SketchQr<CUDA, double, double, double, magma_int_t>;
template class SketchQr<CUDA, double, double, float, magma_int_t>;
template class SketchQr<CUDA, double, float, double, magma_int_t>;
template class SketchQr<CUDA, double, float, float, magma_int_t>;
template class SketchQr<CUDA, float, float, float, magma_int_t>;
template class SketchQr<CUDA, float, __half, float, magma_int_t>;
template class SketchQr<CUDA, double, __half, double, magma_int_t>;
template class SketchQr<CUDA, double, __half, float, magma_int_t>;


}  // end of namespace preconditioner
}  // end of namespace rls
