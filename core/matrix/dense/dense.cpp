#include "../../../include/base_types.hpp"
#include "../../../utils/convert.hpp"
#include "../../../utils/io.hpp"
#include "../memory/magma_context.hpp"
#include "../memory/memory.hpp"
#include "../../blas/blas.hpp"
#include "dense.hpp"


namespace rls {
namespace matrix {


template <ContextType device_type, typename value_type>
std::shared_ptr<Context<device_type>> Dense<device_type, value_type>::get_context()
{
    return Mtx<device_type>::get_context();
}

template <ContextType device_type, typename value_type>
gko::LinOp* Dense<device_type, value_type>::get_mtx()
{
    return Dense<device_type, value_type>::mtx.get();
}

template <ContextType device_type, typename value_type>
std::shared_ptr<gko::LinOp> Dense<device_type, value_type>::get_mtx_shared()
{
    return mtx;
}

template <ContextType device_type, typename value_type>
magma_int_t Dense<device_type, value_type>::get_alloc_elems()
{
    return alloc_elems;
}

template <ContextType device_type, typename value_type>
size_t Dense<device_type, value_type>::get_num_elems()
{
    return this->size_[0] * this->size_[1];
}

template <ContextType device_type, typename value_type>
void Dense<device_type, value_type>::set_matrix(std::shared_ptr<gko::LinOp> mtx_in)
{
    if (std::is_same<value_type, double>::value) {
        mtx = mtx_in;
        auto t = static_cast<gko::matrix::Dense<double>*>(mtx.get());
        values_ = (value_type*)t->get_values();
    }
    else if (std::is_same<value_type, float>::value) {
        mtx = mtx_in;
        auto t = static_cast<gko::matrix::Dense<float>*>(mtx.get());
        values_ = (value_type*)t->get_values();
    }
    this->size_   = {static_cast<int>(mtx->get_size()[0]), static_cast<int>(mtx->get_size()[1])};
    magma_int_t ld_         = mtx->get_size()[0];
    magma_int_t alloc_elems = this->size_[0] * this->size_[1];
    value_type* values_     = nullptr;
}

template <ContextType device_type, typename value_type>
void Dense<device_type, value_type>::zeros()
{
    memory::zeros<value_type, device_type>(this->size_, values_);
}

template <ContextType device_type, typename value_type>
void Dense<device_type, value_type>::eye()
{
    if (std::is_same<value_type, double>::value) {
        memory::eye<double, device_type>({this->size_[0], this->size_[1]}, (double*)values_);
    }
    else if (std::is_same<value_type, float>::value) {
        memory::eye<float, device_type>({this->size_[0], this->size_[1]}, (float*)values_);
    }
}

template <ContextType device_type, typename value_type>
Dense<device_type, value_type>::Dense(std::shared_ptr<Context<device_type>> context,
    dim2 size) : Mtx<device_type>(context)
{
    this->size_ = size;
    this->ld_ = size[0];
    malloc();
    this->alloc_elems = size[0] * size[1];
    if (std::is_same<value_type, __half>::value) {
        cusparseCreateDnMat(&descr_, static_cast<int64_t>(this->size_[0]), static_cast<int64_t>(this->size_[1]),
            static_cast<int64_t>(this->size_[0]), values_, CUDA_R_16F, CUSPARSE_ORDER_COL);
    }
    this->zeros();
}

template <ContextType device_type, typename value_type>
cusparseDnMatDescr_t Dense<device_type, value_type>::get_descriptor()
{
    return descr_;
}

template <ContextType device_type, typename value_type>
Dense<device_type, value_type>::~Dense()
{
    free();
}

template <ContextType device_type, typename value_type>
const value_type* Dense<device_type, value_type>::get_const_values() const { return this->values_; }

template <ContextType device_type, typename value_type>
Dense<device_type, value_type>::Dense(Dense&& mtx) : Mtx<device_type>(mtx.get_context())
{
    auto t = mtx.get_size();
    this->size_    = mtx.get_size();
    this->values_  = mtx.values_;
    this->ld_      = mtx.get_ld();
    auto v = mtx.get_values();
    if (std::is_same<value_type, __half>::value) {
        cusparseCreateDnMat(&descr_, static_cast<int64_t>(this->size_[0]), static_cast<int64_t>(this->size_[1]),
            static_cast<int64_t>(this->size_[0]), values_, CUDA_R_16F, CUSPARSE_ORDER_COL);
    }
    v = nullptr;
}

template <ContextType device_type, typename value_type>
void Dense<device_type, value_type>::malloc()
{
    if (std::is_same<value_type, double>::value) {
        auto context = Mtx<device_type>::get_context();
        auto exec = context->get_executor();
        gko::dim<2> s = {this->size_[0], this->size_[1]};
        mtx = gko::share(gko::matrix::Dense<double>::create(exec, s));
        values_ = (value_type*)(static_cast<gko::matrix::Dense<double>*>(mtx.get())->get_values());
    }
    else if (std::is_same<value_type, float>::value) {
        auto context = Mtx<device_type>::get_context();
        auto exec = context->get_executor();
        gko::dim<2> s = {this->size_[0], this->size_[1]};
        mtx = gko::share(gko::matrix::Dense<float>::create(exec, s));
        values_ = (value_type*)static_cast<gko::matrix::Dense<float>*>(mtx.get())->get_values();
    }
    else if (std::is_same<value_type, __half>::value) {
        this->values_ = nullptr;
        memory::malloc<device_type>(&this->values_, this->size_[0] * this->size_[1]);
    }
}

template <ContextType device_type, typename value_type>
void Dense<device_type, value_type>::free()
{
    if (!wrapper_) {
        if (std::is_same<value_type, __half>::value) {
            cusparseDestroyDnMat(descr_);
            memory::free<device_type>(this->values_);
        }
    }
}

template <ContextType device_type, typename value_type>
Dense<device_type, value_type>::Dense(std::shared_ptr<Context<device_type>> context) : Mtx<device_type>(context) {}


template <ContextType device_type, typename value_type>
Dense<device_type, value_type>::Dense(std::shared_ptr<Context<device_type>> context,
    std::string& filename_mtx) : Mtx<device_type>(context)
{
    io::read_mtx_size((char*)filename_mtx.c_str(), &this->size_[0],
        &this->size_[1]);
    this->ld_ = this->size_[0];
    malloc();
    this->alloc_elems = this->size_[0] * this->size_[1];

    if (std::is_same<value_type, double>::value) {
        io::read_mtx_values<value_type, device_type>(Mtx<device_type>::get_context(),
            (char*)filename_mtx.c_str(), this->size_, this->values_);
    }
    else if (std::is_same<value_type, float>::value) {
        io::read_mtx_values<value_type, device_type>(Mtx<device_type>::get_context(),
            (char*)filename_mtx.c_str(), this->size_, this->values_);
    }
}

template <ContextType device_type, typename value_type>
Dense<device_type, value_type>::Dense(std::shared_ptr<Context<device_type>> context, dim2 size, value_type* values) : Mtx<device_type>(context)
{
    auto exec = context->get_executor();
    wrapper_ = true;
    alloc_elems = size[0] * size[1];
    this->size_ = size;
    this->ld_ = size[0];
    if (std::is_same<value_type, __half>::value) {
        values_ = values;
        cusparseCreateDnMat(&descr_, static_cast<int64_t>(this->size_[0]), static_cast<int64_t>(this->size_[1]),
            static_cast<int64_t>(this->size_[0]), values_, CUDA_R_16F, CUSPARSE_ORDER_COL);
    }
    else if (std::is_same<value_type, double>::value) {
        mtx = gko::matrix::Dense<double>::create(
            exec, gko::dim<2>(size[0], size[1]),
            gko::make_array_view(exec, size[0]*size[1],
                             (double*)values), 1);
        auto t = static_cast<gko::matrix::Dense<double>*>(mtx.get());
        values_ = (value_type*)t->get_values();
    }
    else if (std::is_same<value_type, float>::value) {
        mtx = gko::matrix::Dense<float>::create(
            exec, gko::dim<2>(size[0], size[1]),
            gko::make_array_view(exec, size[0]*size[1],
                             (float*)values), 1);
        auto t = static_cast<gko::matrix::Dense<float>*>(mtx.get());
        values_ = (value_type*)t->get_values();
    }
    values = nullptr;
}

template <ContextType device_type, typename value_type>
Dense<device_type, value_type>::Dense(std::shared_ptr<Context<device_type>> context, dim2 size, magma_int_t ld,
    value_type* values) : Mtx<device_type>(context)
{
    auto exec = context->get_executor();
    wrapper_ = true;
    if (std::is_same<value_type, __half>::value) {
        values_ = values;
        cusparseCreateDnMat(&descr_, static_cast<int64_t>(this->size_[0]), static_cast<int64_t>(this->size_[1]),
            static_cast<int64_t>(this->size_[0]), values_, CUDA_R_16F, CUSPARSE_ORDER_COL);
    }
    else if (std::is_same<value_type, double>::value) {
        mtx = gko::matrix::Dense<double>::create(
            exec, gko::dim<2>(size[0], size[1]),
            gko::make_array_view(exec, size[0]*size[1],
                             (double*)values), ld);
        auto t = static_cast<gko::matrix::Dense<double>*>(mtx.get());
        values_ = (value_type*)t->get_values();
    }
    else if (std::is_same<value_type, float>::value) {
        mtx = gko::matrix::Dense<float>::create(
            exec, gko::dim<2>(size[0], size[1]),
            gko::make_array_view(exec, size[0]*size[1],
                             (float*)values), ld);
        auto t = static_cast<gko::matrix::Dense<float>*>(mtx.get());
        values_ = (value_type*)t->get_values();
    }
    alloc_elems = size[0] * size[1];
    this->size_ = size;
    this->ld_ = ld;
}

template <ContextType device_type, typename value_type>
void Dense<device_type, value_type>::apply(Dense<device_type, value_type>* rhs, Dense<device_type, value_type>* result)
{
    value_type alpha = 1.0;
    value_type beta = 0.0;
    blas::gemm(Mtx<device_type>::get_context(), MagmaNoTrans, MagmaNoTrans, this->size_[0],
          rhs->get_size()[1], this->size_[1], alpha, values_,
          this->size_[0], rhs->get_values(), rhs->get_size()[0],
          beta, result->get_values(), result->get_size()[0]);
}

template <ContextType device_type, typename value_type>
void Dense<device_type, value_type>::apply(value_type alpha, Dense<device_type, value_type>* rhs, value_type beta, Dense<device_type, value_type>* result)
{
    blas::gemm(Mtx<device_type>::get_context(), MagmaNoTrans, MagmaNoTrans, this->size_[0],
          rhs->get_size()[1], this->size_[1], alpha, values_,
          this->size_[0], rhs->get_values(), rhs->get_size()[0],
          beta, result->get_values(), result->get_size()[0]);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create(
    std::shared_ptr<Context<device_type>> context)
{
    auto tmp = new Dense<device_type, value_type>(context);
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create(
    std::shared_ptr<Context<device_type>> context, dim2 size)
{
    auto tmp = new Dense<device_type, value_type>(context, size);
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create(
    std::shared_ptr<Context<device_type>> context, std::string& filename_mtx)
{
    auto tmp = new Dense<device_type, value_type>(context, filename_mtx);
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create(
    std::shared_ptr<Context<device_type>> context, dim2 size, value_type* values)
{
    auto tmp = new Dense<device_type, value_type>(context, size, values);
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create(
    std::shared_ptr<Context<device_type>> context, dim2 size, magma_int_t ld, value_type* values)
{
    auto tmp = new Dense<device_type, value_type>(context, size, ld, values);
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create_submatrix(
    matrix::Dense<device_type, value_type>* mtx_in, span cspan)
{
    auto context = mtx_in->get_context();
    auto size = dim2(mtx_in->get_size()[0], cspan.get_end() - cspan.get_begin() + 1);
    auto tmp = new Dense<device_type, value_type>(context, size, mtx_in->get_values() + mtx_in->get_size()[0] * cspan.get_begin());
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create_subcol(
    matrix::Dense<device_type, value_type>* mtx_in, span rspan, int col)
{
    auto context = mtx_in->get_context();
    auto size = dim2(rspan.get_end() - rspan.get_begin() + 1, 1);
    auto tmp = new Dense<device_type, value_type>(context, size, mtx_in->get_values() + mtx_in->get_size()[0] * col + rspan.get_begin());
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::create_submatrix(
    matrix::Dense<device_type, value_type>* mtx_in, span rspan, span cspan)
{
    auto context = mtx_in->get_context();
    auto size = mtx_in->get_size();
    auto size_submtx = dim2(rspan.get_end() - rspan.get_begin() + 1, cspan.get_end() - cspan.get_begin() + 1);
    auto values = mtx_in->get_values();
    //auto tmp = new Dense<device_type, value_type>(context, size_submtx, mtx_in->get_ld(), values);
    auto tmp = new Dense<device_type, value_type>(context, size_submtx, values);
    // @ERROR, @FIX explicit copying to restore performance matrix issue.
    utils::convert(mtx_in->get_context(), size_submtx[0], size_submtx[1],
        values + rspan.get_begin(), size[0], tmp->get_values(), size[0]);
    return std::unique_ptr<Dense<device_type, value_type>>(tmp);
}

template <ContextType device_type, typename value_type>
template <typename value_type_in>
void Dense<device_type, value_type>::copy_from(matrix::Dense<device_type, value_type_in>* mtx)
{
    auto a = mtx->get_context();
    this->set_context(mtx->get_context());
    this->size_ = mtx->get_size();
    this->ld_   = mtx->get_ld();
    if ((this->alloc_elems == 0) && (this->values_ == nullptr)){
      this->malloc();
    }
    this->alloc_elems = mtx->get_alloc_elems();
    utils::convert(Mtx<device_type>::get_context(), this->size_[0], this->size_[1],
        mtx->get_values(), mtx->get_ld(), values_, this->ld_);

    // @FIX.
    //if ((std::is_same<value_type, double>::value) || (std::is_same<value_type, float>::value)) {
    //    this->mtx = mtx->get_mtx_shared();
    //}
}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::transpose()
{
    auto c = Mtx<device_type>::get_context();
    auto exec = c->get_executor();
    value_type* v = nullptr;
    dim2 s;
    std::shared_ptr<gko::LinOp> t;
    memory::malloc<rls::CUDA>(&v, this->size_[0] * this->size_[1]);
    if (std::is_same<value_type, double>::value) {
        t = rls::share(static_cast<gko::matrix::Dense<double>*>(this->mtx.get())->transpose());
        s = dim2(t->get_size()[0], t->get_size()[1]);
        exec->copy(s[0] * s[1], (value_type*)static_cast<gko::matrix::Dense<double>*>(t.get())->get_values(), v);
    }
    else if (std::is_same<value_type, float>::value) {
        t = rls::share(static_cast<gko::matrix::Dense<float>*>(this->mtx.get())->transpose());
        s = dim2(t->get_size()[0], t->get_size()[1]);
        exec->copy(s[0] * s[1], (value_type*)static_cast<gko::matrix::Dense<float>*>(t.get())->get_values(), v);
    }
    else if (std::is_same<value_type, __half>::value) {
        s = dim2(this->size_[1], this->size_[0]);
        cuda::transpose(this->size_[0], this->size_[1], values_, this->size_[0], v, this->size_[1]);
    }
    auto out = Dense::create(c, s, v);
    return out;
}

//template <ContextType device_type, typename value_type>
//std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::transpose_in_place()
//{
//    auto mtx = gko::matrix::Dense<double>::create(
//        exec, gko::dim<2>(size[0], size[1]),
//        gko::make_array_view(exec, size[0]*size[1],
//                         (double*)values), 1);
//}

template <ContextType device_type, typename value_type>
std::unique_ptr<Dense<device_type, value_type>> Dense<device_type, value_type>::row_to_col_order()
{
    auto c = Mtx<device_type>::get_context();
    value_type* v = nullptr;
    dim2 s;
    std::shared_ptr<gko::LinOp> t;
    if (std::is_same<value_type, double>::value) {
        t = static_cast<gko::matrix::Dense<double>*>(this->mtx.get())->transpose();
        v = (value_type*)static_cast<gko::matrix::Dense<double>*>(t.get())->get_values();
    }
    else if (std::is_same<value_type, float>::value) {
        t = static_cast<gko::matrix::Dense<float>*>(this->mtx.get())->transpose();
        v = (value_type*)static_cast<gko::matrix::Dense<float>*>(t.get())->get_values();
    }
    else if (std::is_same<value_type, __half>::value) {
        memory::malloc<rls::CUDA>(&v, this->size_[0] * this->size_[1]);
        cuda::transpose(this->size_[0], this->size_[1], values_, this->size_[0], v, this->size_[1]);
    }
    return Dense::create(c, dim2(this->get_size()[0], this->get_size()[1]), std::move(v));
}

template<> std::unique_ptr<Dense<rls::CUDA, __half>> Dense<rls::CUDA, __half>::transpose()
{
    auto c = this->get_context();
    __half* v = nullptr;
    dim2 s;
    std::shared_ptr<gko::LinOp> t;
    memory::malloc<rls::CUDA>(&v, this->size_[0] * this->size_[1]);
    s = dim2(this->size_[1], this->size_[0]);
    cuda::transpose(this->size_[0], this->size_[1], values_, this->size_[0], v, this->size_[1]);
    return Dense<rls::CUDA, __half>::create(c, s, std::move(v));
}

template class Dense<CUDA, double>;
template class Dense<CUDA, float>;
template class Dense<CUDA, __half>;
template class Dense<CPU, double>;
template class Dense<CPU, float>;

template void Dense<CUDA, double>::copy_from(matrix::Dense<CUDA, double>* mtx);
template void Dense<CUDA, double>::copy_from(matrix::Dense<CUDA, float>* mtx);
template void Dense<CUDA, float>::copy_from(matrix::Dense<CUDA, float>* mtx);
template void Dense<CUDA, float>::copy_from(matrix::Dense<CUDA, double>* mtx);

template void Dense<CPU, double>::copy_from(matrix::Dense<CPU, double>* mtx);
template void Dense<CPU, double>::copy_from(matrix::Dense<CPU, float>* mtx);
template void Dense<CPU, float>::copy_from(matrix::Dense<CPU, float>* mtx);
template void Dense<CPU, float>::copy_from(matrix::Dense<CPU, double>* mtx);


template void Dense<CUDA, __half>::copy_from(matrix::Dense<CUDA, __half>* mtx);
template void Dense<CUDA, __half>::copy_from(matrix::Dense<CUDA, float>* mtx);
template void Dense<CUDA, __half>::copy_from(matrix::Dense<CUDA, double>* mtx);
template void Dense<CUDA, float>::copy_from(matrix::Dense<CUDA, __half>* mtx);
template void Dense<CUDA, double>::copy_from(matrix::Dense<CUDA, __half>* mtx);

template <ContextType device_type, typename value_type>
Dense<device_type, value_type>& Dense<device_type, value_type>::operator=(Dense<device_type, value_type>&& mtx_in)
{
    this->set_context(mtx_in.get_context());
    this->ld_ = mtx_in.get_ld();
    this->alloc_elems = mtx_in.get_alloc_elems();
    this->size_ = mtx_in.get_size();
    this->values_ = mtx_in.get_values();
    this->mtx = mtx_in.get_mtx_shared();
    return *this;
}


} // end of namespace matrix
} // end of namespace rls
