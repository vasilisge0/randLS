#include "generalized_split.hpp"
#include "../matrix/dense/dense.hpp"
#include "../../utils/io.hpp"


namespace rls {
namespace preconditioner {
namespace generalized_split {

template <typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
Config<value_type, value_type_internal_0, value_type_precond_0, index_type>::Config(double sampling_coefficient)
{
    this->sampling_coefficient_ = sampling_coefficient;
}

template <typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
std::unique_ptr<Config<value_type, value_type_internal_0, value_type_precond_0, index_type>>
    Config<value_type, value_type_internal_0, value_type_precond_0, index_type>::create(double sampling_coefficient)
{
    return std::unique_ptr<Config<value_type, value_type_internal_0, value_type_precond_0, index_type>>
        (new Config<value_type, value_type_internal_0, value_type_precond_0, index_type>(sampling_coefficient));
}


template <typename value_type, typename value_type_internal_0,
          typename index_type>
void state<value_type, value_type_internal_0, index_type>::allocate(index_type ld_mtx, index_type num_cols_mtx,
              index_type num_rows_sketch, index_type num_cols_sketch,
              index_type ld_sketch, index_type ld_r_factor)
{
    memory::malloc<CUDA>(&dmtx_rp, ld_mtx * num_cols_mtx);
    memory::malloc<CUDA>(&dsketch_rp, ld_sketch * num_cols_sketch);
    memory::malloc<CUDA>(&dresult_rp, ld_r_factor * num_cols_mtx);
    memory::malloc<CPU>(&tau, num_rows_sketch);
}

template <typename value_type, typename value_type_internal_0,
          typename index_type>
void state<value_type, value_type_internal_0, index_type>::free()
{
    memory::free<CUDA>(dmtx_rp);
    memory::free<CUDA>(dsketch_rp);
    memory::free<CUDA>(dresult_rp);
    memory::free<CPU>(tau);
}

template class Config<double, double, double, magma_int_t>;
template class Config<double, float, double, magma_int_t>;
template class Config<double, __half, double, magma_int_t>;
template class Config<float, double, double, magma_int_t>;
template class Config<float, __half, double, magma_int_t>;
//template class GeneralizedSplit<CUDA, double, double, __half, magma_int_t>;
template class Config<double, float, float, magma_int_t>;
template class Config<float, float, float, magma_int_t>;
template class Config<float, __half, float, magma_int_t>;


} // end of generalized_split

template <ContextType device_type, typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
void GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>::apply(
    std::shared_ptr<Context<device_type>> context, magma_trans_t trans, matrix::Dense<device_type, value_type>* u_vector)
{
    auto size = dim2(u_vector->get_size()[0] - precond_mtx_->get_size()[1], precond_mtx_->get_size()[1]);
    //auto u1 = rls::matrix::Dense<device_type, value_type>::create(context,
    //    dim2(size[1], 1), u_vector->get_values() + size[0]);
        std::cout << "before u1\n";
    //auto u1 = matrix::Dense<device_type, value_type>::create_submatrix(u_vector,
    //    span(size[0], size[0] + size[1]), span(0, 0));
    auto u1 = matrix::Dense<device_type, value_type>::create_subcol(u_vector,
        span(size[0], size[0] + size[1]), 0);

    temp1->copy_from(u1.get());
    value_type_precond_0 one = 1.0;
    value_type_precond_0 zero = 0.0;
    blas::gemv(context, trans, size[1], size[1], one, precond_mtx_internal_->get_values(),
           precond_mtx_->get_size()[0], temp1->get_values(), 1, zero, temp2->get_values(), 1);
    u1->copy_from(temp2.get());
}

template <ContextType device_type, typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
void GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>::generate()
{
    auto context = this->get_context();
    auto mtx = this->get_mtx();
    auto logger = this->get_logger();
    auto size = mtx->get_size();
    auto sampled_rows = sketch_->get_size()[0];
    auto precond_state = new state<device_type, value_type, value_type_internal_0, index_type>(mtx.get(), mtx->get_size(), sampled_rows, sampled_rows, mtx->get_size()[0]);
    index_type info_qr = 0;
    auto queue = context->get_queue();
    auto t = static_cast<matrix::Sparse<device_type, value_type_precond_0, index_type>*>(mtx.get());
    info_qr = sketch_qr_impl(context, sketch_, mtx, precond_mtx_, precond_state);
    if (info_qr != 0) {
        magma_xerbla("geqrf2_gpu", info_qr);
    }
    blas::trtri(context, MagmaUpper, MagmaNonUnit, precond_mtx_->get_size()[1],
                precond_mtx_->get_values(), precond_mtx_->get_ld(), &info_qr);
    // Needs to transpose precond_mtx_->get_values()
    precond_mtx_internal_->copy_from(precond_mtx_.get());
}

template <ContextType device_type, typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
template<typename value_type_out>
std::shared_ptr<GeneralizedSplit<device_type, value_type_out, value_type_out, value_type_out, index_type>> GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>::convert_to()
{
    auto tmp = rls::share(GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_out, index_type>::create());
    auto mtx = this->get_mtx();
    tmp->mtx_->copy_from(mtx);
    tmp->precond_mtx_->copy_from(precond_mtx_);
    return tmp;
}

template <ContextType device_type, typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
std::unique_ptr<
    GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>>
GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>::create(std::shared_ptr<MtxOp<device_type>> mtx,
       std::shared_ptr<SketchOperator<device_type, value_type_internal_0, index_type>> sketch,
       std::shared_ptr<preconditioner::Config> config)
{
    auto t = std::static_pointer_cast<generalized_split::Config<value_type, value_type_internal_0, value_type_precond_0, index_type>>(config);
    return std::unique_ptr<
        GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>>(
        new GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>(mtx,
            sketch, t));
}

template <ContextType device_type, typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
matrix::Dense<device_type, value_type>* GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>::get_precond()
{
    return precond_mtx_.get();
}

template <ContextType device_type, typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>::GeneralizedSplit(
       std::shared_ptr<MtxOp<device_type>> mtx,
       std::shared_ptr<SketchOperator<device_type, value_type_internal_0, index_type>> sketch,
       std::shared_ptr<generalized_split::Config<value_type, value_type_internal_0, value_type_precond_0, index_type>> config) : PrecondOperator<device_type, value_type, index_type>(mtx)
{
    config_ = config;
    auto context = this->get_context();
    this->sketch_ = sketch;
    this->precond_mtx_ = matrix::Dense<device_type, value_type>::create(context, {sketch->get_size()[0], mtx->get_size()[1]});
    this->precond_mtx_internal_ = matrix::Dense<device_type, value_type_precond_0>::create(context, {sketch->get_size()[0], mtx->get_size()[1]});
    std::cout << "constructor generalized split\n";
    std::cout << "mtx->get_size()[1]: " << mtx->get_size()[1] << '\n';
    temp1 = matrix::Dense<device_type, value_type_precond_0>::create(context, {mtx->get_size()[1], 1});
    temp2 = matrix::Dense<device_type, value_type_precond_0>::create(context, {mtx->get_size()[1], 1});
}

template class GeneralizedSplit<CUDA, double, double, double, magma_int_t>;
template class GeneralizedSplit<CUDA, double, float, double, magma_int_t>;
template class GeneralizedSplit<CUDA, double, __half, double, magma_int_t>;
template class GeneralizedSplit<CUDA, double, __half, float, magma_int_t>;
template class GeneralizedSplit<CUDA, float, double, double, magma_int_t>;
template class GeneralizedSplit<CUDA, double, double, __half, magma_int_t>;
template class GeneralizedSplit<CUDA, double, float, float, magma_int_t>;
template class GeneralizedSplit<CUDA, float, float, float, magma_int_t>;
template class GeneralizedSplit<CUDA, float, __half, float, magma_int_t>;
template class GeneralizedSplit<CUDA, double, float, __half, magma_int_t>;
template class GeneralizedSplit<CUDA, double, __half, __half, magma_int_t>;


}   // end of namespace preconditioner
}   // end of namespace rls
