#include "../../include/base_types.hpp"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../cuda/sketch/countsketch.cuh"
#include "../matrix/dense/dense.hpp"
#include "../matrix/sparse/sparse.hpp"
#include "../blas/blas.hpp"
#include "../memory/magma_context.hpp"
#include "gaussian.hpp"
#include "countsketch.hpp"


namespace rls {


template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
CountSketch<device_type, value_type, value_type_apply, index_type>::CountSketch(std::shared_ptr<Context<device_type>> context, size_t k, std::string& filename_mtx)
{
    this->context_ = context;
    this->nnz_per_col_ = k;
    auto mtx = rls::share(matrix::Sparse<device_type, value_type, index_type>::create(
        context, filename_mtx));
    auto size = mtx->get_size();
    this->size_ = size;
    this->mtx_ = matrix::Sparse<device_type, value_type_apply, index_type>::create(
        context, size, size[1]);
    convert_to(size[1], mtx.get(), this->mtx_.get());
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
CountSketch<device_type, value_type, value_type_apply, index_type>::CountSketch(std::shared_ptr<Context<device_type>> context, size_t k, dim2 size)
{
    this->context_ = context;
    this->nnz_per_col_ = k;
    this->size_ = size;
    auto mtx = rls::share(matrix::Sparse<device_type, value_type, index_type>::create(context, {size[1], size[0]}, k*size[1]));
    auto t = rls::share(matrix::Sparse<device_type, value_type_apply, index_type>::create(context, {size[1], size[0]}, k*size[1]));
    sketch::countsketch_impl(nnz_per_col_, mtx);
    convert_to(k, mtx.get(), t.get());
    this->mtx_ = t->transpose();
}

template<> CountSketch<rls::CUDA, double, __half, magma_int_t>::CountSketch(std::shared_ptr<Context<rls::CUDA>> context, size_t k, dim2 size)
{
    this->context_ = context;
    this->nnz_per_col_ = k;
    this->size_ = size;
    auto mtx = rls::share(matrix::Sparse<rls::CUDA, double, magma_int_t>::create(context, {size[1], size[0]}, k*size[1]));
    auto t = rls::share(matrix::Sparse<rls::CUDA, __half, magma_int_t>::create(context, {size[1], size[0]}, k*size[1]));
    sketch::countsketch_impl(nnz_per_col_, mtx);
    t->copy_from(mtx.get());
    this->mtx_ = t->transpose();
}

template<>
CountSketch<rls::CUDA, double, __half, magma_int_t>::CountSketch(std::shared_ptr<Context<rls::CUDA>> context, size_t k, std::string& filename_mtx)
{
    //this->context_ = context;
    //this->nnz_per_col_ = k;
    //auto mtx = rls::share(matrix::Sparse<device_type, value_type, index_type>::create(
    //    context, filename_mtx));
    //auto size = mtx->get_size();
    //this->mtx_ = matrix::Sparse<device_type, value_type_apply, index_type>::create(
    //    context, size, size[1]);
    //convert_to(size[1], mtx.get(), this->mtx_.get());
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
std::unique_ptr<CountSketch<device_type, value_type, value_type_apply, index_type>>
        CountSketch<device_type, value_type, value_type_apply, index_type>::create(std::shared_ptr<Context<device_type>> context, size_t k, dim2 size)
{
    using SketchType = CountSketch<device_type, value_type, value_type_apply, index_type>;
    return std::unique_ptr<SketchType>(
        new CountSketch(context, k, size));
}

template<> std::unique_ptr<CountSketch<rls::CUDA, double, __half, magma_int_t>>
        CountSketch<rls::CUDA, double, __half, magma_int_t>::create(std::shared_ptr<Context<rls::CUDA>> context, size_t k, dim2 size)
{
    using SketchType = CountSketch<rls::CUDA, double, __half, magma_int_t>;
    return std::unique_ptr<SketchType>(
        new CountSketch(context, k, size));
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
std::unique_ptr<
    CountSketch<device_type, value_type, value_type_apply, index_type>>
        CountSketch<device_type, value_type, value_type_apply, index_type>::create(std::shared_ptr<Context<device_type>> context, size_t k,
               std::string& filename_mtx)
{
    using SketchType = CountSketch<device_type, value_type, value_type_apply, index_type>;
    return std::unique_ptr<SketchType>(
        new CountSketch(context, k, filename_mtx));
}

template<>
std::unique_ptr<
    CountSketch<rls::CUDA, double, __half, magma_int_t>>
        CountSketch<rls::CUDA, double, __half, magma_int_t>::create(std::shared_ptr<Context<rls::CUDA>> context, size_t k,
               std::string& filename_mtx)
{
    using SketchType = CountSketch<rls::CUDA, double, __half, magma_int_t>;
    return std::unique_ptr<SketchType>(
        new CountSketch(context, k, filename_mtx));
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
value_type_apply* CountSketch<device_type, value_type, value_type_apply, index_type>::get_values()
{
    return mtx_->get_values();
}

template<> __half* CountSketch<rls::CUDA, double, __half, magma_int_t>::get_values()
{
    return mtx_->get_values();
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
dim2 CountSketch<device_type, value_type, value_type_apply, index_type>::get_size()
{
    auto size = mtx_->get_size();
    dim2 s = {size_[0], size_[1]};
    return s;
}

template<> dim2 CountSketch<rls::CUDA, double, __half, magma_int_t>::get_size()
{
    return size_;
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
void CountSketch<device_type, value_type, value_type_apply, index_type>::set_mtx(std::shared_ptr<matrix::Sparse<device_type, value_type_apply, index_type>> t)
{
    mtx_ = t;
}

template<> void CountSketch<rls::CUDA, double, __half, magma_int_t>::set_mtx(std::shared_ptr<matrix::Sparse<rls::CUDA, __half, magma_int_t>> t)
{
    mtx_ = t;
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
std::shared_ptr<matrix::Sparse<device_type, value_type_apply, index_type>> CountSketch<device_type, value_type, value_type_apply, index_type>::get_mtx()
{
    return mtx_;
}

template<> std::shared_ptr<matrix::Sparse<rls::CUDA, __half, magma_int_t>> CountSketch<rls::CUDA, double, __half, magma_int_t>::get_mtx()
{
    return mtx_;
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
CountSketch<device_type, value_type, value_type_apply, index_type>::CountSketch(CountSketch<device_type, value_type, value_type_apply, index_type> &t)
{
    this->mtx_ = t.get_mtx();
}

template<> CountSketch<rls::CUDA, double, __half, magma_int_t>::CountSketch(CountSketch<rls::CUDA, double, __half, magma_int_t> &t)
{
    this->mtx_ = t.get_mtx();
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
CountSketch<device_type, value_type, value_type_apply, index_type>& CountSketch<device_type, value_type, value_type_apply, index_type>::operator=(CountSketch<device_type, value_type, value_type_apply, index_type>& t)
{
    this->set_mtx(t.get_mtx());
    return *this;
}

// this needs fixing
template<> CountSketch<rls::CUDA, double, __half, magma_int_t>& CountSketch<rls::CUDA, double, __half, magma_int_t>::operator=(CountSketch<rls::CUDA, double, __half, magma_int_t>& t)
{
    //this->set_mtx(t.get_mtx());
    return *this;
}

//
template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
size_t CountSketch<device_type, value_type, value_type_apply, index_type>::get_nnz_per_col()
{
    return nnz_per_col_;
}


// @ok
template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
void CountSketch<device_type, value_type, value_type_apply, index_type>::convert_to(
    int k,
    matrix::Sparse<device_type, value_type, index_type>* mtx_in,
    matrix::Sparse<device_type, value_type_apply, index_type>* mtx_out)
{
    auto size = mtx_in->get_size();
    utils::convert(this->context_, k*size[0], 1, mtx_in->get_values(),
                   k*size[0], mtx_out->get_values(), k*size[0]);
    auto context = mtx_out->get_context();
    utils::convert(this->context_, static_cast<int>(size[0] + 1), 1, mtx_in->get_row_ptrs(),
                   static_cast<int>(size[0] + 1), mtx_out->get_row_ptrs(), static_cast<int>(size[0] + 1));
    utils::convert(this->context_, k*size[0], 1, mtx_in->get_col_idxs(),
                   k*size[0], mtx_out->get_col_idxs(), k*size[0]);
}

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
void CountSketch<device_type, value_type, value_type_apply, index_type>::apply(
    std::shared_ptr<MtxOp<device_type>> rhs,
    std::shared_ptr<matrix::Dense<device_type, value_type_apply>> result)
{
    if (auto t = dynamic_cast<matrix::Dense<device_type, value_type_apply>*>(rhs.get()); t != nullptr) {
        auto queue = this->context_->get_queue();
        value_type_apply one  = 1.0;
        value_type_apply zero = 0.0;
        auto size = this->get_size();
        const auto sketch = this->mtx_->get_values();
        auto sketch_rows = this->mtx_->get_size()[0];
        auto rhs_cols = rhs->get_size()[1];
        auto rhs_rows = rhs->get_size()[0];
        size_t str = rhs_cols;
        auto exec = this->context_->get_executor();
        gko::dim<2> s1 = {rhs_rows, rhs_cols};
        auto rhs_tmp = gko::share(gko::matrix::Dense<value_type_apply>::create(exec, s1));
        dim2 s2 = {rhs_rows, rhs_cols};  // doesn't work

        // (!!) change to value_type_apply
        rls::memory::zeros<value_type, rls::CUDA>(s2, (value_type*)rhs_tmp->get_values());
        // (!!) ok here is the problem. FIXED
        rls::cuda::transpose(rhs->get_size()[0], rhs->get_size()[1], t->get_values(), rhs->get_size()[0], rhs_tmp->get_values(), rhs->get_size()[1]);

        gko::dim<2> s3 = {sketch_rows, rhs_cols};
        auto sol = gko::share(gko::matrix::Dense<value_type_apply>::create(exec, s3, gko::make_array_view(exec, rhs_cols * rhs_rows, result->get_values()), str));
        auto mtx = mtx_->get_mtx();
        dim2 s4 = {static_cast<int>(s3[0]), static_cast<int>(s3[1])};
        rls::memory::zeros<value_type, rls::CUDA>(s4, (value_type*)sol->get_values());
        mtx->apply(rhs_tmp, sol);
        rls::cuda::transpose(s4[0], s4[1], sol->get_values(), s4[0], result->get_values(), s4[1]);
    }
    else if (auto t = dynamic_cast<matrix::Sparse<device_type, value_type_apply, index_type>*>(rhs.get()); t != nullptr) {
        auto queue = this->context_->get_queue();
        value_type_apply one  = 1.0;
        value_type_apply zero = 0.0;
        auto size = this->get_size();
        const auto sketch = this->mtx_->get_values();
        auto sketch_rows = this->mtx_->get_size()[0];
        auto rhs_cols = rhs->get_size()[1];
        auto rhs_rows = rhs->get_size()[0];
        size_t str = rhs_cols;
        auto exec = this->context_->get_executor();
        gko::dim<2> s1 = {rhs_rows, rhs_cols};
        auto rhs_tmp = t->get_mtx();
        gko::dim<2> s3 = {sketch_rows, rhs_cols};
        dim2 s4 = {sketch_rows, rhs_cols};
        // replace this with sparse
        auto SOL = rls::share(rls::matrix::Sparse<device_type, value_type_apply, index_type>::create(this->context_, s4)); //  , gko::make_array_view(exec, rhs_cols * rhs_rows, result->get_values()))
        mtx_->apply(t, SOL.get());
        auto s = rls::share(rls::matrix::Dense<device_type, value_type_apply>::create(this->context_, s4));
        SOL->to_dense(s.get());
        auto s_trans = rls::share(s->row_to_col_order());
        result->copy_from(s_trans.get());
    }
}

template<> void CountSketch<rls::CUDA, double, __half, magma_int_t>::apply(
    std::shared_ptr<MtxOp<rls::CUDA>> rhs,
    std::shared_ptr<matrix::Dense<rls::CUDA, __half>> result)
{
std::cout << "SKETCH APPLY\n";
    if (auto t = dynamic_cast<matrix::Dense<rls::CUDA, __half>*>(rhs.get()); t != nullptr) {
        auto queue = this->context_->get_queue();
        __half one  = 1.0;
        __half zero = 0.0;
        auto size = this->get_size();
        const auto sketch = this->mtx_->get_values();
        auto sketch_rows = this->mtx_->get_size()[0];
        auto rhs_cols = rhs->get_size()[1];
        auto rhs_rows = rhs->get_size()[0];
        size_t str = rhs_cols;
        auto exec = this->context_->get_executor();
        gko::dim<2> s1 = {rhs_rows, rhs_cols};
        //auto rhs_tmp = gko::share(gko::matrix::Dense<__half>::create(exec, s1));
        //dim2 s2 = {rhs_rows, rhs_cols};  // doesn't work

        //rls::memory::zeros<__half, rls::CUDA>(s2, (__half*)rhs_tmp->get_values());
        //// (!!) ok here is the problem. FIXED
        //rls::cuda::transpose(rhs->get_size()[0], rhs->get_size()[1], t->get_values(), rhs->get_size()[0], rhs_tmp->get_values(), rhs->get_size()[1]);

    //    gko::dim<2> s3 = {sketch_rows, rhs_cols};
    //    auto sol = gko::share(gko::matrix::Dense<value_type_apply>::create(exec, s3, gko::make_array_view(exec, rhs_cols * rhs_rows, result->get_values()), str));
    //    auto mtx = mtx_->get_mtx();
    //    dim2 s4 = {static_cast<int>(s3[0]), static_cast<int>(s3[1])};
    //    rls::memory::zeros<value_type, rls::CUDA>(s4, (value_type*)sol->get_values());
    //    mtx->apply(rhs_tmp, sol);
    //    rls::cuda::transpose(s4[0], s4[1], sol->get_values(), s4[0], result->get_values(), s4[1]);
    }
    else if (auto t = dynamic_cast<matrix::Sparse<rls::CUDA, __half, magma_int_t>*>(rhs.get()); t != nullptr) {
        auto queue = this->context_->get_queue();
        __half one  = 1.0;
        __half zero = 0.0;
        auto size = this->get_size();
        const auto sketch = this->mtx_->get_values();
        auto sketch_rows = size_[0];
        auto rhs_cols = t->get_size()[1];
        //auto rhs_rows = rhs->get_size()[0];
        auto rhs_rows = 1;
        size_t str = rhs_cols;
        auto exec = this->context_->get_executor();
        gko::dim<2> s1 = {rhs_rows, rhs_cols};
        auto rhs_tmp = t->get_mtx();
        gko::dim<2> s3 = {sketch_rows, rhs_cols};
        dim2 s4 = {sketch_rows, rhs_cols};
        auto SOL = rls::share(rls::matrix::Sparse<rls::CUDA, __half, magma_int_t>::create(this->context_, s4));
        // set nonzeros to SOL
        mtx_->apply(t, SOL.get());
        auto T = rls::matrix::Sparse<rls::CUDA, double, int>::create(context_, SOL->get_size(),
            SOL->get_nnz());
        T->copy_from(SOL.get());

        //auto s = rls::share(rls::matrix::Dense<rls::CUDA, __half>::create(this->context_, s4));
        auto ts = dim2(size[0], SOL->get_size()[1]);
        //auto s = rls::share(rls::matrix::Dense<rls::CUDA, __half>::create(this->context_, ts));
        SOL->to_dense(result.get());
        //result->copy_from(s.get());
        //auto T = rls::matrix::Dense<rls::CUDA, double>::create(context_, result->get_size());
        //T->copy_from(result.get());
//@FIX
        //auto s_trans = rls::share(s->transpose());
        //result->copy_from(s_trans.get());
    }
}


template class SketchOperator<CUDA, double, magma_int_t>;
template class SketchOperator<CUDA, float, magma_int_t>;
template class SketchOperator<CUDA, __half, magma_int_t>;

template class CountSketch<CUDA, double, double, magma_int_t>;
template class CountSketch<CUDA, double, float, magma_int_t>;
//template class CountSketch<CUDA, double, __half, magma_int_t>;
template class CountSketch<CUDA, float, float, magma_int_t>;
template class CountSketch<CUDA, double, __half, magma_int_t>;
//template class CountSketch<CUDA, float, __half, magma_int_t>;


}   // end of namespace rls
