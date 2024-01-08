//#include "../memory/memory.hpp"
//#include "preconditioner.hpp"
#include "gaussian.hpp"
#include "../blas/blas.hpp"
#include "../memory/magma_context.hpp"
#include "../matrix/sparse/sparse.hpp"
#include "../../include/base_types.hpp"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../utils/io.hpp"
#include "../../utils/convert.hpp"


namespace rls {


template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
std::unique_ptr<GaussianSketch<device_type, value_type, value_type_apply, index_type>> GaussianSketch<device_type, value_type, value_type_apply, index_type>::create(std::shared_ptr<Context<device_type>> context, dim2 size)
{
    return std::unique_ptr<GaussianSketch<device_type, value_type, value_type_apply, index_type>>(new GaussianSketch(context, size));
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
std::unique_ptr<GaussianSketch<device_type, value_type, value_type_apply, index_type>> GaussianSketch<device_type, value_type, value_type_apply, index_type>::create(
    std::shared_ptr<Context<device_type>> context, std::string& filename_mtx)
{
    return std::unique_ptr<GaussianSketch<device_type, value_type, value_type_apply, index_type>>(new GaussianSketch(context, filename_mtx));
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
dim2 GaussianSketch<device_type, value_type, value_type_apply, index_type>::get_size()
{
    return mtx_->get_size();
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
value_type_apply* GaussianSketch<device_type, value_type, value_type_apply, index_type>::get_values()
{
    return mtx_->get_values();
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
std::shared_ptr<matrix::Dense<device_type, value_type_apply>> GaussianSketch<device_type, value_type, value_type_apply, index_type>::get_mtx()
{
    return mtx_;
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
void GaussianSketch<device_type, value_type, value_type_apply, index_type>::convert_to(
    matrix::Dense<device_type, value_type>* mtx_in,
    matrix::Dense<device_type, value_type_apply>* mtx_out)
{
    auto size = mtx_in->get_size();
    utils::convert(this->context_, size[0], size[1], mtx_in->get_values(), mtx_in->get_ld(),
                   mtx_out->get_values(), mtx_out->get_ld());
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
void GaussianSketch<device_type, value_type, value_type_apply, index_type>::apply(std::shared_ptr<MtxOp<device_type>> rhs, std::shared_ptr<matrix::Dense<device_type, value_type_apply>> result)
{
    if (auto t = dynamic_cast<matrix::Dense<device_type, value_type_apply>*>(rhs.get()); t != nullptr) {
        auto queue = this->mtx_->get_context()->get_queue();
        value_type_apply one  = 1.0;
        value_type_apply zero = 0.0;
        auto size = this->mtx_->get_size();
        const auto sketch = this->mtx_->get_values();
        const auto rhs_tmp = t->get_values();
        auto sketch_rows = this->mtx_->get_size()[0];
        auto rhs_cols = rhs->get_size()[1];
        auto rhs_rows = rhs->get_size()[0];
        blas::gemm(this->mtx_->get_context(), MagmaNoTrans, MagmaNoTrans, sketch_rows, rhs_cols,
                   rhs_rows, one, sketch, this->mtx_->get_ld(), rhs_tmp,
                   t->get_ld(), zero, result->get_values(), result->get_ld());
    }
    else if (auto t = dynamic_cast<matrix::Sparse<device_type, value_type_apply, index_type>*>(rhs.get()); t != nullptr) {
    std::cout << ">>>test\n";
        auto context = mtx_->get_context();
        auto exec = context->get_executor();
        value_type_apply one = 1.0;
        value_type_apply zero = 0.0;
        auto A = t->transpose();
        auto S = mtx_->transpose();
        auto t1_size = dim2(static_cast<int>(A->get_size()[0]), static_cast<int>(S->get_size()[1]));
        auto t1 = matrix::Dense<device_type, value_type_apply>::create(context, t1_size);
        std::cout << "before apply" << '\n';
        std::cout << "sizeof(mtx_->get_values()): " << sizeof(mtx_->get_values()) << '\n';
        std::cout << "sizeof(value_type_apply): " << sizeof(value_type_apply) << '\n';
        A->apply(one, S.get(), zero, t1.get());
        // this works also
        //exec->copy(t1->get_size()[0] * t1->get_size()[1], t1->get_values(),
        //    result->get_values());
        std::cout << "t1->get_size(){0]: " << t1->get_size()[0] << ", t1->get_size()[1]: " << t1->get_size()[1] << ", t1->get_ld(): " << t1->get_ld() << '\n';
        utils::convert(context,
            (int)(t1->get_size()[0] * t1->get_size()[1]),
            1,
            t1->get_values(),
            (int)(t1->get_size()[0] * t1->get_size()[1]),
            result->get_values(),
            (int)(t1->get_size()[0] * t1->get_size()[1]));
    }
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
GaussianSketch<device_type, value_type, value_type_apply, index_type>::GaussianSketch(std::shared_ptr<Context<device_type>> context, dim2 size)
{
    this->context_ = context;
    auto mtx = rls::share(matrix::Dense<device_type, double>::create(context, size));
    this->mtx_ = matrix::Dense<device_type, value_type_apply>::create(context, size);
    sketch::gaussian_sketch_impl(mtx);
    mtx_->copy_from(mtx.get());
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
GaussianSketch<device_type, value_type, value_type_apply, index_type>::GaussianSketch(std::shared_ptr<Context<device_type>> context,
    std::string& filename_mtx)
{
    this->context_ = context;
    auto mtx = rls::share(matrix::Dense<device_type, value_type>::create(context, filename_mtx));
    auto size = mtx->get_size();
    mtx_ = matrix::Dense<device_type, value_type_apply>::create(context, size);
    mtx_->copy_from(mtx.get());
}

//template class SketchOperator<CUDA, double, magma_int_t>;
//template class SketchOperator<CUDA, float, magma_int_t>;
//template class SketchOperator<CUDA, __half, magma_int_t>;

template class GaussianSketch<CUDA, double, double, magma_int_t>;
template class GaussianSketch<CUDA, double, float, magma_int_t>;
template class GaussianSketch<CUDA, double, __half, magma_int_t>;
template class GaussianSketch<CUDA, float, float, magma_int_t>;
//template class GaussianSketch<CUDA, float, __half, magma_int_t>;


}   // end of namespace rls
