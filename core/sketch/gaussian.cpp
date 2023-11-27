//#include "../memory/memory.hpp"
//#include "preconditioner.hpp"
#include "gaussian.hpp"
#include "../blas/blas.hpp"
#include "../memory/magma_context.hpp"
#include "../matrix/sparse/sparse.hpp"
#include "../../include/base_types.hpp"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../utils/io.hpp"


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
        auto context = mtx_->get_context();
        auto exec = context->get_executor();
        //auto one = gko::initialize<gko::matrix::Dense<value_type>>(
        //    {(value_type)1.0}, exec);
        //auto zero = gko::initialize<gko::matrix::Dense<value_type>>(
        //    {(value_type)0.0}, exec);
        //mtx_->get_mtx()->apply(one, t->get_mtx(), zero, result->get_mtx());
        //// Fix this
        //auto A = static_cast<gko::matrix::Csr<value_type_apply, index_type>*>(t->get_mtx().get())->transpose();
        //auto S = static_cast<gko::matrix::Dense<value_type_apply>*>(mtx_->get_mtx())->transpose();
        //auto t1 = gko::matrix::Dense<value_type_apply>::create(exec, gko::dim<2>(A->get_size()[0], S->get_size()[1]));
        value_type one = 1.0;
        value_type zero = 0.0;
        std::cout << "<here>\n";

        auto A = t->transpose();
        auto S = mtx_->transpose();

        std::cout << "S: \n";
        io::scan_for_nan_gpu(context, S->get_size()[0], S->get_size()[1],
            S->get_values(), S->get_size()[0]);

        std::cout << "mtx_->get_size()[0]: " << mtx_->get_size()[0] << "\n";
        auto T = mtx_->get_mtx();
        std::cout << "T->get_size()[0]: " << T->get_size()[0] << '\n';

        //{
        //    auto queue = context->get_queue();
        //    std::cout << "\n\n\n before mm:\n";
        //    std::cout << "mtx_\n";
        //    io::scan_for_nan_gpu(context, mtx_->get_size()[0], mtx_->get_size()[1], mtx_->get_values(),
        //        mtx_->get_ld());

        //    std::cout << "s\n";
        //    io::scan_for_nan_gpu(context, S->get_size()[0], S->get_size()[1], S->get_values(),
        //        S->get_ld());

        //    std::cout << "mtx_\n";
        //    io::scan_for_nan_gpu(context, mtx_->get_size()[0], mtx_->get_size()[1], mtx_->get_values(),
        //        mtx_->get_ld());

        //    std::cout << "s\n";
        //    io::scan_for_nan_gpu(context, S->get_size()[0], S->get_size()[1], S->get_values(),
        //        S->get_ld());

        //    std::cout << "t\n";
        //    io::scan_for_nan_gpu(context, t->get_nnz(), 1, t->get_values(),
        //        t->get_nnz());

        //    std::cout << "A\n";
        //    io::scan_for_nan_gpu(context, A->get_nnz(), 1, A->get_values(),
        //        A->get_nnz());
        //    //io::write_mtx("S4.mtx", t->get_nnz(), 1,
        //    //    (float*)t->get_values(), t->get_nnz(), queue);
        //}

        ////auto A1 = matrix::Sparse<device_type, value_type_apply, index_type>::create(context,
        ////    A->get_size(), A->get_nnz());
        ////auto S1 = matrix::Dense<device_type, value_type_apply>::create(context,
        ////    S->get_size());
        //std::cout << "mtx_->get_size()[0]: " << mtx_->get_size()[0] << '\n';
        //std::cout << "mtx_->get_size()[1]: " << mtx_->get_size()[1] << '\n';
        //std::cout << "t->get_size()[0]: " << t->get_size()[0] << '\n';
        //std::cout << "t->get_size()[1]: " << t->get_size()[1] << '\n';
        //std::cout << "A->get_size()[0]: " << A->get_size()[0] << '\n';
        //std::cout << "A->get_size()[1]: " << A->get_size()[1] << '\n';
        //std::cout << "S->get_size()[0]: " << S->get_size()[0] << '\n';
        //std::cout << "S->get_size()[1]: " << S->get_size()[1] << '\n';
        //std::cout << "S->get_ld(): " << S->get_ld() << '\n';
        auto t1_size = dim2(static_cast<int>(A->get_size()[0]), static_cast<int>(S->get_size()[1]));
        auto t1 = matrix::Dense<device_type, value_type_apply>::create(context, t1_size);
        //std::cout << "t1->get_size()[0]: " << t1->get_size()[0] << ", t1->get_size()[1]: " << t1->get_size()[1] << '\n';
        //std::cout << "t1->get_ld(): " << t1->get_ld() << '\n';
        //std::cout << "before apply\n";

    {
        ////auto precond_mtx_ = precond_mtx;
        ////io::write_mtx("precond_hgdp.mtx", S->get_size()[0], S->get_size()[1],
        ////    (double*)S->get_values(), S->get_ld(), queue);
    }
        //cudaDeviceSynchronize();
        A->apply(one, S.get(), zero, t1.get());
        //cudaDeviceSynchronize();

        ////A1->apply(one, S.get(), zero, t1.get());
        ////A1->apply(one, S1.get(), zero, t1.get());
        //std::cout << "after apply\n";
        //exec->copy(t1->get_size()[0] * t1->get_size()[1], t1->get_values(),
        //    result->get_values());

        //{
        //    std::cout << "\n\nbefore: \n";
        //    auto queue = context->get_queue();
        //    io::print_mtx_gpu(4, 4, result->get_values(), result->get_ld(), queue);
        //}

        //{
        //    auto queue = context->get_queue();
        //    std::cout << "before copy\n";
        //    //io::scan_for_nan_gpu(context, result->get_size()[0], result->get_size()[1],
        //    //    result->get_values(), result->get_ld());
        //    io::scan_for_nan_gpu(context, t1->get_size()[0], t1->get_size()[1],
        //        t1->get_values(), t1->get_size()[0]);
        //}

std::cout << "result->get_size()[0]: " << result->get_size()[0] << '\n';
std::cout << "result->get_ld(): " << result->get_ld() << '\n';
        //cuda::transpose(t1->get_size()[0], t1->get_size()[1], t1->get_values(),
        //    t1->get_ld(), result->get_values(), result->get_ld());
    }
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
GaussianSketch<device_type, value_type, value_type_apply, index_type>::GaussianSketch(std::shared_ptr<Context<device_type>> context, dim2 size)
{
    this->context_ = context;
    auto mtx = rls::share(matrix::Dense<device_type, double>::create(context, size));
    this->mtx_ = matrix::Dense<device_type, value_type_apply>::create(context, size);
    std::cout << "sizeof(value_type): " << sizeof(value_type) << '\n';
    sketch::gaussian_sketch_impl(mtx);
    std::cout << "in copy\n";
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
