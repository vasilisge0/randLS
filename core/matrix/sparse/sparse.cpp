#include "../memory/magma_context.hpp"
#include "../memory/memory.hpp"
#include "dense.hpp"
#include "sparse.hpp"
#include "../../../utils/convert.hpp"


namespace rls {
namespace matrix {

template <ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>& Sparse<device_type, value_type, index_type>::operator=(Sparse<device_type, value_type, index_type>& mtx_in)
{
std::cout << "in operator=\n";
    nnz_ = mtx_in.get_nnz();
    size_ = mtx_in.get_size();
    values_ = std::move(mtx_in.get_values());
    row_ptrs_ = std::move(mtx_in.get_row_ptrs());
    col_idxs_ = std::move(mtx_in.get_col_idxs());
    mtx_  = mtx_in.get_mtx();
    descr_ = mtx_in.get_descriptor();
    return *this;
}

template <ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>& Sparse<device_type, value_type, index_type>::operator=(Sparse<device_type, value_type, index_type>&& mtx_in)
{
std::cout << "in move operator=\n";
    nnz_ = mtx_in.get_nnz();
    size_ = mtx_in.get_size();

    if (values_ != nullptr) {
        memory::free<rls::CUDA>(values_);
    }
    values_ = mtx_in.values_;
    mtx_in.values_ = nullptr;

    if (row_ptrs_ != nullptr) {
        memory::free<rls::CUDA>(row_ptrs_);
    }
    row_ptrs_ = mtx_in.row_ptrs_;
    mtx_in.row_ptrs_ = nullptr;

    if (row_ptrs_ != nullptr) {
        memory::free<rls::CUDA>(col_idxs_);
    }
    col_idxs_ = mtx_in.col_idxs_;
    mtx_in.col_idxs_ = nullptr;

    mtx_  = mtx_in.mtx_;
    mtx_in.mtx_ = nullptr;
    descr_ = mtx_in.descr_;
    mtx_in.descr_ = nullptr;
    return *this;
}


template<ContextType device_type, typename value_type, typename index_type>
dim2 Sparse<device_type, value_type, index_type>::get_size()
{
    auto t = this->get_mtx();
    dim2 s = {static_cast<int>(t->get_size()[0]), static_cast<int>(t->get_size()[1])};
    return s;
}

template<> dim2 Sparse<rls::CUDA, __half, magma_int_t>::get_size() {
    return size_;
}

template<ContextType device_type, typename value_type, typename index_type>
size_t Sparse<device_type, value_type, index_type>::get_nnz()
{
    return nnz_;
}

template <ContextType device_type, typename value_type, typename index_type>
std::shared_ptr<Context<device_type>> Sparse<device_type, value_type, index_type>::get_context()
{
    return Mtx<device_type>::get_context();
}

template<ContextType device_type, typename value_type, typename index_type>
value_type* Sparse<device_type, value_type, index_type>::get_values()
{
    using matrix = gko::matrix::Csr<value_type, index_type>;
    auto t = static_cast<matrix*>(this->mtx_.get());
    return t->get_values();
}

template<> __half* Sparse<rls::CUDA, __half, magma_int_t>::get_values()
{
    return values_;
}

template<ContextType device_type, typename value_type, typename index_type>
index_type* Sparse<device_type, value_type, index_type>::get_row_ptrs()
{
    using matrix = gko::matrix::Csr<value_type, index_type>;
    auto t = static_cast<matrix*>(this->mtx_.get());
    return t->get_row_ptrs();
}

template<> magma_int_t* Sparse<rls::CUDA, __half, magma_int_t>::get_row_ptrs()
{
    return row_ptrs_;
}

template<ContextType device_type, typename value_type, typename index_type>
index_type* Sparse<device_type, value_type, index_type>::get_col_idxs()
{
    using matrix = gko::matrix::Csr<value_type, index_type>;
    auto t = static_cast<matrix*>(this->mtx_.get());
    return t->get_col_idxs();
}

template<> magma_int_t* Sparse<rls::CUDA, __half, magma_int_t>::get_col_idxs()
{
    return col_idxs_;
}

//template<>
//class Sparse<CUDA, double, magma_int_t>;
//
//template<>
//class Sparse<CUDA, float, magma_int_t>;

//template <ContextType device_type, typename value_type, typename index_type>
//Sparse<device_type, value_type>& Sparse<device_type, value_type>::operator=(Sparse<device_type, value_type>&& mtx)
//{
//    if (*this != mtx) {
//        this->mtx_ = mtx.get_mtx();
//    }
//    return *this;
//}

//template <ContextType device_type, typename value_type, typename index_type>
//Sparse<device_type, value_type>& Sparse<device_type, value_type>:copy_from(std::shared_ptr<Mtx<device_type>> mtx)
//{
//
//}

//template <ContextType device_type, typename value_type, typename index_type>
//template<typename value_type_in>
//void Sparse<device_type, value_type>::copy_from(matrix::Sparse<device_type, value_type_in, index_type>* mtx)

template <ContextType device_type, typename value_type, typename index_type>
template<typename value_type_in>
void Sparse<device_type, value_type, index_type>::copy_from(matrix::Sparse<device_type, value_type_in, index_type>* mtx)
{
    this->set_context(mtx->get_context());
    auto s = mtx->get_size();
    this->nnz_ = mtx->get_nnz();
    utils::convert(this->get_context(), static_cast<int>(this->nnz_), 1, mtx->get_values(), static_cast<int>(this->nnz_), this->get_values(), static_cast<int>(this->nnz_));
    auto context = this->get_context();
    auto exec = context->get_executor();
    exec->copy(s[0] + 1, mtx->get_row_ptrs(), this->get_row_ptrs());
    exec->copy(this->nnz_, mtx->get_col_idxs(), this->get_col_idxs());
}

template<ContextType device_type, typename value_type, typename index_type>
cusparseSpMatDescr_t Sparse<device_type, value_type, index_type>::get_descriptor()
{
    return descr_;
}

template<ContextType device_type, typename value_type, typename index_type>
void Sparse<device_type, value_type, index_type>::set_row_ptrs(index_type* row_ptrs)
{
    row_ptrs_ = row_ptrs;
}

template<ContextType device_type, typename value_type, typename index_type>
void Sparse<device_type, value_type, index_type>::set_col_idxs(index_type* col_idxs)
{
    col_idxs_ = col_idxs;
}

template<ContextType device_type, typename value_type, typename index_type>
void Sparse<device_type, value_type, index_type>::set_values(value_type* values)
{
    values_ = values;
}

template<ContextType device_type, typename value_type, typename index_type>
void Sparse<device_type, value_type, index_type>::set_descriptor(cusparseSpMatDescr_t descr)
{
    descr_ = descr;
}

template<ContextType device_type, typename value_type, typename index_type>
void Sparse<device_type, value_type, index_type>::apply(Sparse<device_type, value_type, index_type>* rhs, Sparse<device_type, value_type, index_type>* result)
{
    mtx_->apply(rhs->get_mtx(), result->get_mtx());
    auto context = this->get_context();
    auto exec = context->get_executor();
    auto sol = static_cast<gko::matrix::Csr<value_type, index_type>*>(result->get_mtx().get());
    nnz_ = exec->copy_val_to_host(sol->get_row_ptrs() + sol->get_size()[0]);
}

//template<> void Sparse<rls::CUDA, double, magma_int_t>::apply(Sparse<rls::CUDA, double, magma_int_t>* rhs,
//    Sparse<rls::CUDA, double, magma_int_t>* result)
//{
//    mtx_->apply(rhs->get_mtx(), result->get_mtx());
//}
//
//template<> void Sparse<rls::CUDA, float, magma_int_t>::apply(Sparse<rls::CUDA, float, magma_int_t>* rhs,
//    Sparse<rls::CUDA, float, magma_int_t>* result)
//{
//    mtx_->apply(rhs->get_mtx(), result->get_mtx());
//}A
//


template<ContextType device_type, typename value_type, typename index_type>
void Sparse<device_type, value_type, index_type>::apply(value_type alpha, Dense<device_type, value_type>* rhs, value_type beta, Dense<device_type, value_type>* result)
{
    auto context = this->get_context();
    auto exec = context->get_executor();
    auto alpha_mtx = gko::initialize<gko::matrix::Dense<value_type>>(
        {alpha}, exec);
    auto beta_mtx = gko::initialize<gko::matrix::Dense<value_type>>(
        {beta}, exec);
    cudaDeviceSynchronize();
    auto t0 = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get());
    auto t1 = static_cast<gko::matrix::Dense<value_type>*>(rhs->get_mtx());
    auto t2 = static_cast<gko::matrix::Dense<value_type>*>(result->get_mtx());
    t0->apply(alpha_mtx.get(), t1, beta_mtx.get(), t2);
    //static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get())->apply(alpha_mtx.get(), static_cast<gko::matrix::Dense<value_type>*>(rhs->get_mtx()), beta_mtx.get(),
    //    static_cast<gko::matrix::Dense<value_type>*>(result->get_mtx()));
    //cudaDeviceSynchronize();

    //auto t = static_cast<gko::matrix::Dense<value_type>*>(rhs->get_mtx());
    auto t = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get());
    {
        auto queue = context->get_queue();
        std::cout << "result->get_ld(): " << result->get_ld() << '\n';
        std::cout << "result->get_size()[0]: " << result->get_size()[0] << '\n';
        //io::write_mtx("S8.mtx", result->get_size()[0], result->get_size()[1],
        //    (float*)result->get_values(), result->get_ld(), queue);
        //io::write_mtx("S8.mtx", t->get_size()[0], t->get_size()[1],
        //    (float*)t->get_values(), t->get_size()[0], queue);
        //io::write_mtx("S8.mtx", this->get_nnz(), 1,
        //    (float*)t->get_values(), this->get_nnz(), queue);
        //std::cout << "t2->get_size()[1]: " << t2->get_size()[1] << '\n';
        std::cout << "<-t2->\n";
        std::cout << "t2->get_size()[0]: " << t2->get_size()[0] << '\n';
        std::cout << "t2->get_size()[1]: " << t2->get_size()[1] << '\n';
        //io::write_mtx("S2.mtx", t2->get_size()[0], t2->get_size()[1],
        //    (float*)t2->get_values(), t2->get_size()[0], queue);
        //io::write_mtx("S2.mtx", t2->get_size()[0]*t2->get_size()[1], 1,
        //    (float*)t2->get_values(), t2->get_size()[0]*t2->get_size()[1], queue);
    }
}

template<> void Sparse<rls::CUDA, __half, magma_int_t>::apply(__half alpha, Dense<rls::CUDA, half>* rhs, __half beta, Dense<rls::CUDA, __half>* result)
{
    auto context = this->get_context();
    auto exec = context->get_executor();
    //auto alpha_mtx = gko::initialize<gko::matrix::Dense<value_type>>(
    //    {alpha}, exec);
    //auto beta_mtx = gko::initialize<gko::matrix::Dense<value_type>>(
    //    {beta}, exec);
    //static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get())->apply(alpha_mtx.get(), static_cast<gko::matrix::Dense<value_type>*>(rhs->get_mtx()), beta_mtx.get(),
    //    static_cast<gko::matrix::Dense<value_type>*>(result->get_mtx()));
}

template<ContextType device_type, typename value_type, typename index_type>
void Sparse<device_type, value_type, index_type>::to_dense(Dense<device_type, value_type>* result)
{
    static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get())->convert_to(
        static_cast<gko::matrix::Dense<value_type>*>(result->get_mtx()));
}

template<> void Sparse<rls::CUDA, __half, magma_int_t>::to_dense(Dense<rls::CUDA, __half>* result)
{
std::cout << "\n\nIN TO DENSE\n\n";
    auto context = this->get_context();
    auto cusparse_handle = context->get_cusparse_handle();
    std::cout << "here\n";
    cusparseDnMatDescr_t descr_dense = result->get_descriptor();
    //cusparseDnMatDescr_t descr_dense;
    cusparseSparseToDenseAlg_t alg = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
    size_t buffer_size = 0;
    std::cout <<"(values_ == nullptr): " << (values_ == nullptr) << '\n';
    cusparseCreateCsr(&descr_, size_[0], size_[1], nnz_, row_ptrs_, col_idxs_, values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnMat(&descr_dense, static_cast<int64_t>(size_[0]), static_cast<int64_t>(size_[1]),
        static_cast<int64_t>(size_[0]), result->get_values(), CUDA_R_16F, CUSPARSE_ORDER_COL);
    cusparseSparseToDense_bufferSize(cusparse_handle, descr_, descr_dense, alg, &buffer_size);
    void* buffer = nullptr;
    std::cout << "buffer_size: " << buffer_size << '\n';
    cudaMalloc(&buffer, buffer_size);
    cusparseSparseToDense(cusparse_handle, descr_, descr_dense, alg, buffer);
    cudaFree(buffer);

    //auto T = rls::matrix::Dense<rls::CUDA, double>::create(context, result->get_size());
    //T->copy_from(result);
    //auto queue = context->get_queue();
    //std::cout << "result->get_values(): \n";
    //rls::io::print_mtx_gpu(5, 1, T->get_values(), T->get_size()[0], queue);

    //int64_t rows;
    //int64_t cols;
    //int64_t ld;
    ////auto v = result->get_values();
    //void* v;
    ////auto tmp = matrix::Dense<rls::CUDA, __half>::create(context);
    //cudaDataType type;
    //cusparseOrder_t order;
    //std::cout << "ttttttttt\n";
    //cusparseDnMatGet(
    //    descr_dense,
    //    &rows,
    //    &cols,
    //    &ld,
    //    (void**)&v,
    //    &type,
    //    &order);
    //utils::convert(context, (int)(rows*cols), (int)1, (__half*)v,
    //    (int)(rows*cols), result->get_values(), (int)(rows*cols));
    //std::cout << "order: " << order << '\n';
    //cusparseDestroyDnMat(descr_dense);
}

template<ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::Sparse(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz,
       value_type* values, index_type* row_ptrs, index_type* col_idxs) : Mtx<device_type>(context)
{
    nnz_ = nnz;
    size_ = size;
    values_ = values;
    row_ptrs_ = row_ptrs;
    col_idxs_ = col_idxs;
    auto exec = context->get_executor();
    auto s = gko::dim<2>(size[0], size[1]);
    auto r = gko::make_array_view(exec, size[0] + 1, row_ptrs_);
    auto c = gko::make_array_view(exec, nnz, col_idxs);
    auto v = gko::make_array_view(exec, nnz, values);
    mtx_ = gko::share(gko::matrix::Csr<value_type, index_type>::create(exec, s, v, c, r));
}

template<>
Sparse<rls::CUDA, __half, magma_int_t>::Sparse(std::shared_ptr<Context<rls::CUDA>> context, dim2 size, size_t nnz,
       __half* values, magma_int_t* row_ptrs, magma_int_t* col_idxs) : Mtx<rls::CUDA>(context)
{
    nnz_ = nnz;
    size_ = size;
    values_ = values;
    row_ptrs_ = row_ptrs;
    col_idxs_ = col_idxs;
}

template<ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::Sparse(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz) : Mtx<device_type>(context)
{
    auto exec = context->get_executor();
    gko::dim<2> s = {static_cast<int>(size[0]), static_cast<int>(size[1])};
    this->size_ = size; // new
    this->nnz_ = nnz;
    mtx_ = gko::share(gko::matrix::Csr<value_type, index_type>::create(exec, s, nnz));
    using matrix = gko::matrix::Csr<value_type, index_type>;
    auto t = static_cast<matrix*>(this->mtx_.get());
    values_ = t->get_values();
    row_ptrs_ = t->get_row_ptrs();
    col_idxs_ = t->get_col_idxs();
}

// create
template<> Sparse<rls::CUDA, __half, magma_int_t>::Sparse(std::shared_ptr<Context<rls::CUDA>> context, dim2 size, size_t nnz) : Mtx<rls::CUDA>(context)
{
    auto exec = context->get_executor();
    gko::dim<2> s = {static_cast<int>(size[0]), static_cast<int>(size[1])};
    this->size_ = size; // new
    this->nnz_ = nnz;
    memory::malloc<CUDA>(&values_, nnz);
    memory::malloc<CUDA>(&row_ptrs_, s[0] + 1);
    memory::malloc<CUDA>(&col_idxs_, nnz);
    cusparseCreateCsr(&descr_, s[0], s[1], nnz_, row_ptrs_, col_idxs_, values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
}

template<ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::Sparse(std::shared_ptr<Context<device_type>> context,
    std::string& filename_mtx) : Mtx<device_type>(context)
{
    using matrix = gko::matrix::Csr<value_type, index_type>;
    auto exec = context->get_executor();
    mtx_ = gko::share(gko::read<matrix>(std::ifstream(filename_mtx), exec));
    auto queue = context->get_queue();
    // Need to compute nnz here
    auto s = mtx_->get_size();
    index_type nnz = 0;
    auto R = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get());
    nnz_ = static_cast<size_t>(exec->copy_val_to_host(&(R->get_row_ptrs()[s[0]])));
    size_ = dim2(mtx_->get_size()[0], mtx_->get_size()[1]);
    values_ = R->get_values();
    row_ptrs_ = R->get_row_ptrs();
    col_idxs_ = R->get_col_idxs();
}

template<> Sparse<rls::CUDA, __half, magma_int_t>::Sparse(std::shared_ptr<Context<rls::CUDA>> context,
    std::string& filename_mtx) : Mtx<rls::CUDA>(context)
{
    //using matrix = gko::matrix::Csr<value_type, index_type>;
    auto exec = context->get_executor();
    //mtx_ = gko::share(gko::read<matrix>(std::ifstream(filename_mtx), exec));
    auto queue = context->get_queue();
    // Need to compute nnz here
    auto s = mtx_->get_size();
    magma_int_t nnz = 0;
    //auto R = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get());
    //nnz_ = static_cast<size_t>(exec->copy_val_to_host(&(R->get_row_ptrs()[s[0]])));
}

template<ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::Sparse(std::shared_ptr<Context<device_type>> context,
    std::shared_ptr<gko::LinOp> mtx) : Mtx<device_type>(context)
{
    //this->mtx_ = std::shared_ptr<gko::LinOp>(mtx);
    this->mtx_ = mtx;
    auto exec = context->get_executor();
    //this->nnz_ = exec->copy_val_to_host(&static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx.get())->get_row_ptrs()[mtx->get_size()[0]]);
    this->nnz_ = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx.get())->get_num_stored_elements();
    size_ = dim2(mtx->get_size()[0], mtx->get_size()[1]);
    auto R = static_cast<gko::matrix::Csr<value_type, index_type>*>(this->mtx_.get());
    values_ = R->get_values();
    row_ptrs_ = R->get_row_ptrs();
    col_idxs_ = R->get_col_idxs();
}

// @to_impl_later
template<> Sparse<rls::CUDA, __half, magma_int_t>::Sparse(std::shared_ptr<Context<rls::CUDA>> context, std::shared_ptr<gko::LinOp> mtx) : Mtx<rls::CUDA>(context)
{
    //this->mtx_ = std::shared_ptr<gko::LinOp>(mtx);
    auto exec = context->get_executor();
    //this->nnz_ = exec->copy_val_to_host(static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx.get())->get_row_ptrs() + mtx->get_size()[0]);
    //values_ = mmalloc<CUDA>(&values_, size_t n)
}

template<ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::Sparse(std::shared_ptr<Context<device_type>> context, dim2 size) : Mtx<device_type>(context)
{
    auto exec = context->get_executor();
    gko::dim<2> s = {static_cast<int>(size[0]), static_cast<int>(size[1])};
    mtx_ = gko::share(gko::matrix::Csr<value_type, index_type>::create(exec, s)); //  , gko::make_array_view(exec, rhs_cols * rhs_rows, result->get_values()))
    auto R = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get());
    nnz_ = static_cast<size_t>(exec->copy_val_to_host(&(R->get_row_ptrs()[s[0]])));
    size_ = size;
    values_ = R->get_values();
    row_ptrs_ = R->get_row_ptrs();
    col_idxs_ = R->get_col_idxs();
}

//ok
template<> Sparse<rls::CUDA, __half, magma_int_t>::Sparse(std::shared_ptr<Context<rls::CUDA>> context, dim2 size) : Mtx<rls::CUDA>(context)
{
    auto exec = context->get_executor();
    gko::dim<2> s = {static_cast<int>(size[0]), static_cast<int>(size[1])};
    nnz_ = 0;
    size_ = size;
    cusparseCreateCsr(&descr_, s[0], s[1], nnz_, row_ptrs_, col_idxs_, values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
}

template<ContextType device_type, typename value_type, typename index_type>
size_t Sparse<device_type, value_type, index_type>::get_num_elems()
{
    return nnz_;
}

template<ContextType device_type, typename value_type, typename index_type>
std::unique_ptr<Sparse<device_type, value_type, index_type>> Sparse<device_type, value_type, index_type>::create(std::shared_ptr<Context<device_type>> context, dim2 size)
{
    return std::unique_ptr<Sparse<device_type, value_type, index_type>>(new Sparse<device_type, value_type, index_type>(context, size));
}

template<ContextType device_type, typename value_type, typename index_type>
std::unique_ptr<Sparse<device_type, value_type, index_type>> Sparse<device_type, value_type, index_type>::create(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz)
{
    return std::unique_ptr<Sparse<device_type, value_type, index_type>>(new Sparse<device_type, value_type, index_type>(context, size, nnz));
}

template<ContextType device_type, typename value_type, typename index_type>
std::unique_ptr<Sparse<device_type, value_type, index_type>> Sparse<device_type, value_type, index_type>::create(std::shared_ptr<Context<device_type>> context,
    std::string& filename_mtx)
{
    return std::unique_ptr<Sparse<device_type, value_type, index_type>>(new Sparse<device_type, value_type, index_type>(context, filename_mtx));
}

template<ContextType device_type, typename value_type, typename index_type>
static std::unique_ptr<Sparse<device_type, value_type, index_type>> Sparse<device_type, value_type, index_type>::create(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz,
    value_type* values, index_type* row_ptrs, index_type* col_idxs)
{
    return std::unique_ptr<Sparse<device_type, value_type, index_type>>(new Sparse<device_type, value_type, index_type>(context, size, nnz, values, col_idxs, row_ptrs));
}

template<ContextType device_type, typename value_type, typename index_type>
std::unique_ptr<Sparse<device_type, value_type, index_type>> Sparse<device_type, value_type, index_type>::create(std::shared_ptr<Context<device_type>> context,
    std::shared_ptr<gko::LinOp> mtx)
{
    return std::unique_ptr<Sparse<device_type, value_type, index_type>>(new Sparse<device_type, value_type, index_type>(context, mtx));
}

template<ContextType device_type, typename value_type, typename index_type>
std::shared_ptr<gko::LinOp> Sparse<device_type, value_type, index_type>::get_mtx()
{
    return mtx_;
}

template<ContextType device_type, typename value_type, typename index_type>
gko::LinOp* Sparse<device_type, value_type, index_type>::get_raw_mtx()
{
    return mtx_.get();
}

//template<ContextType device_type, typename value_type, typename index_type>
//std::shared_ptr<Context<device_type>> Sparse<device_type, value_type, index_type>::get_context()
//{
//    return this->get_context_;
//}


template<ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::~Sparse() {}

template<> Sparse<rls::CUDA, __half, magma_int_t>::~Sparse() {
std::cout << "in sparse destructor\n";
    if (values_ != nullptr) {
        memory::free<rls::CUDA>(values_);
    }

    if (row_ptrs_ != nullptr) {
        memory::free<rls::CUDA>(row_ptrs_);
    }

    if (row_ptrs_ != nullptr) {
        memory::free<rls::CUDA>(col_idxs_);
    }
std::cout << "out\n";
}

template<ContextType device_type, typename value_type, typename index_type>
std::unique_ptr<Sparse<device_type, value_type, index_type>> Sparse<device_type, value_type, index_type>::transpose()
{
    std::shared_ptr<gko::LinOp> mtx = static_cast<gko::matrix::Csr<value_type, index_type>*>(this->mtx_.get())->transpose();
    auto T = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx.get());
    auto context = this->get_context();
    auto queue = context->get_queue();
    std::cout << "sizeof(T->get_values()): " << sizeof(T->get_values()) << '\n';
    std::cout << "before writting S1\n";
    //io::write_mtx("S1.mtx", this->get_nnz(), 1,
    //    (double*)T->get_values(), this->get_nnz(), queue);
    //io::write_mtx("S11.mtx", this->get_nnz(), 1,
    //    (float*)T->get_values(), this->get_nnz(), queue);

    return Sparse::create(this->get_context(), mtx);
}

template<ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::Sparse(Sparse<device_type, value_type, index_type>&& t) : Mtx<device_type>(t.get_context())
{
    auto context = this->get_context();
    mtx_ = t.get_mtx();
    auto R = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get());
    auto s = R->get_size();
    auto exec = context->get_executor();
    nnz_ = static_cast<size_t>(exec->copy_val_to_host(&(R->get_row_ptrs()[s[0]])));
    values_ = R->get_values();
    row_ptrs_ = R->get_row_ptrs();
    col_idxs_ = R->get_col_idxs();
}

//ok
template<>
Sparse<rls::CUDA, __half, magma_int_t>::Sparse(Sparse<rls::CUDA, __half, magma_int_t>&& t) : Mtx<rls::CUDA>(t.get_context())
{
    //context_ = t.get_context();
    //mtx_ = t.get_mtx();
   // auto R = static_cast<gko::matrix::Csr<value_type, index_type>*>(mtx_.get());
   auto tmp = t.get_row_ptrs();
    //nnz_ = static_cast<size_t>(exec->copy_val_to_host(&(R->get_row_ptrs()[s[0]])));
    //values_ = R->get_values();
    //row_ptrs_ = R->get_row_ptrs();
    //col_idxs_ = R->get_col_idxs();
}



template<>
template<>
void Sparse<rls::CUDA, double, magma_int_t>::copy_from(matrix::Sparse<rls::CUDA, __half, magma_int_t>* mtx)
{
    this->set_context(mtx->get_context());
    auto s = mtx->get_size();
    this->nnz_ = mtx->get_nnz();
    utils::convert(this->get_context(), static_cast<int>(this->nnz_), 1, mtx->get_values(),
        static_cast<int>(this->nnz_), this->get_values(), static_cast<int>(this->nnz_));
    auto context = this->get_context();
    auto exec = context->get_executor();
    exec->copy(s[0] + 1, mtx->get_row_ptrs(), this->get_row_ptrs());
    exec->copy(this->nnz_, mtx->get_col_idxs(), this->get_col_idxs());
}

template<>
template<>
void Sparse<rls::CUDA, __half, magma_int_t>::copy_from(matrix::Sparse<rls::CUDA, __half, magma_int_t>* mtx)
{
    this->set_context(mtx->get_context());
    auto s = mtx->get_size();
    this->nnz_ = mtx->get_nnz();
    // !!! see what to do here.
    utils::convert(this->get_context(), static_cast<int>(this->nnz_), 1, mtx->get_values(), static_cast<int>(this->nnz_), this->get_values(), static_cast<int>(this->nnz_));
    auto context = this->get_context();
    auto exec = context->get_executor();
    exec->copy(s[0] + 1, mtx->get_row_ptrs(), this->get_row_ptrs());
    exec->copy(this->nnz_, mtx->get_col_idxs(), this->get_col_idxs());
}

template<>
template<>
void Sparse<rls::CUDA, __half, magma_int_t>::copy_from(matrix::Sparse<rls::CUDA, double, magma_int_t>* mtx)
{
    this->set_context(mtx->get_context());
    auto s = mtx->get_size();
    this->nnz_ = mtx->get_nnz();
    utils::convert(this->get_context(), static_cast<int>(this->nnz_), 1, mtx->get_values(), static_cast<int>(this->nnz_), this->get_values(), static_cast<int>(this->nnz_));
    auto context = this->get_context();
    auto exec = context->get_executor();
    exec->copy(s[0] + 1, mtx->get_row_ptrs(), this->get_row_ptrs());
    exec->copy(this->nnz_, mtx->get_col_idxs(), this->get_col_idxs());
}

// to_impl
template<> std::unique_ptr<Sparse<rls::CUDA, __half, magma_int_t>> Sparse<rls::CUDA, __half, magma_int_t>::transpose()
{
    //std::shared_ptr<gko::LinOp> mtx = static_cast<gko::matrix::Csr<value_type, index_type>*>(this->mtx_.get())->transpose();
    //return Sparse::create(context_, mtx);
    auto context = this->get_context();
    auto t = Sparse<rls::CUDA, __half, magma_int_t>::create(context, dim2(size_[1], size_[0]), nnz_);
    auto cusparse_handle = context->get_cusparse_handle();
    int m = size_[0];
    int n = size_[1];
    auto val_csr = values_;
    auto row_ptrs = row_ptrs_;
    auto col_idxs = col_idxs_;
    auto val_csc = t->get_values();
    magma_int_t* col_ptrs_csc = t->get_row_ptrs();
    magma_int_t* row_idxs_csc = t->get_col_idxs();
    cudaDataType val_type = CUDA_R_16F;
    cusparseAction_t copy_values = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idx_base = CUSPARSE_INDEX_BASE_ZERO;
    cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG_DEFAULT;
    size_t buffer_size;

    cusparseCsr2cscEx2_bufferSize(
        cusparse_handle,
        m,
        n,
        nnz_,
        val_csr,
        row_ptrs,
        col_idxs,
        val_csc,
        col_ptrs_csc,
        row_idxs_csc,
        val_type,
        copy_values,
        idx_base,
        alg,
        &buffer_size
    );

    void* buffer = nullptr;
    cudaMalloc(&buffer, buffer_size);
    cusparseCsr2cscEx2(
        cusparse_handle,
        m,
        n,
        nnz_,
        val_csr,
        row_ptrs,
        col_idxs,
        val_csc,
        col_ptrs_csc,
        row_idxs_csc,
        val_type,
        copy_values,
        idx_base,
        alg,
        buffer
    );
    cudaFree(buffer);
    return t;
}

template<> void Sparse<rls::CUDA, __half, magma_int_t>::apply(Sparse<rls::CUDA, __half, magma_int_t>* rhs,
    Sparse<rls::CUDA, __half, magma_int_t>* result)
{
    cusparseSpGEMMDescr_t spgemm_descr;
    cusparseSpGEMM_createDescr(&spgemm_descr);
    __half alpha = 1.0;
    __half beta = 0.0;
    auto context = this->get_context();
    auto cusparse_handle = context->get_cusparse_handle();
    size_t bufferSize1;
    __half* externalBuffer1;
    cusparseSpMatDescr_t descr0;
    cusparseSpMatDescr_t descr1;
    cusparseCreateCsr(&descr0, size_[0], size_[1], nnz_, row_ptrs_, col_idxs_, values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    auto s1 = rhs->get_size();
    auto nnz1 = rhs->get_nnz();
    std::cout << "size_[0]: " << size_[0] << ", size_[1]: " << size_[1] << '\n';
    std::cout << "s1[0]: " << s1[0] << ", s1[1]: " << s1[1] << '\n';
    std::cout << "nnz1: " << nnz1 << ", (int64_t)nnz1: " << (int64_t)nnz1 << '\n';
    std::cout << "(rhs->get_row_ptrs() == nullptr): " << (rhs->get_row_ptrs() == nullptr) << '\n';
    cusparseCreateCsr(&descr1, (int64_t)s1[0], (int64_t)s1[1], (int64_t)nnz1, rhs->get_row_ptrs(), rhs->get_col_idxs(), rhs->get_values(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    auto descr_out = result->get_descriptor();
    //cusparseSpMatDescr_t descr_out;
std::cout << "before work estimation\n";

    int64_t rows, cols, nnz;
    __half* values;
    magma_int_t* row_ptrs;
    magma_int_t* col_idxs;
std::cout << "here\n";
std::cout << "after\n";

    auto status = cusparseSpGEMM_workEstimation(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        descr0,
        descr1,
        &beta,
        descr_out,
        CUDA_R_16F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr,
        &bufferSize1,
        NULL
    );
    std::cout << "status: " << status << '\n';
    std::cout << "<OUT\n";
    cudaMalloc(&externalBuffer1, bufferSize1);
    cusparseSpGEMM_workEstimation(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        descr_,
        rhs->get_descriptor(),
        &beta,
        descr_out,
        CUDA_R_16F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr,
        &bufferSize1,
        externalBuffer1
    );
std::cout << "after work estimation\n";

    size_t bufferSize2;
    size_t bufferSize3;
    __half* externalBuffer2;
    __half* externalBuffer3;
    cusparseSpGEMM_compute(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        descr_,
        rhs->get_descriptor(),
        &beta,
        descr_out,
        CUDA_R_16F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr,
        &bufferSize2,
        NULL);
    cudaMalloc(&externalBuffer2, bufferSize2);
    cusparseSpGEMM_compute(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        descr_,
        rhs->get_descriptor(),
        &beta,
        descr_out,
        CUDA_R_16F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr,
        &bufferSize2,
        externalBuffer2);

    cusparseSpMatGetSize(descr_out, &rows, &cols, &nnz);
    auto size_tmp = dim2(rows, cols);
    memory::malloc<CUDA>(&values, nnz);
    memory::malloc<CUDA>(&row_ptrs, size_[0] + 1);
    memory::malloc<CUDA>(&col_idxs, nnz);
    cusparseCreateCsr(&descr_out, size_[0], s1[1], nnz, row_ptrs, col_idxs, values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    // allocate csr arrays
    cusparseSpGEMM_copy(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        descr_,
        rhs->get_descriptor(),
        &beta,
        descr_out,
        CUDA_R_16F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr);

    auto tmp = rls::matrix::Sparse<rls::CUDA, __half, magma_int_t>::create(context, size_tmp, nnz,
        values, col_idxs, row_ptrs);
std::cout << "\n\n_------- MOVE \n\n";
    //*result = *(tmp.get());
    *result = std::move(*(tmp.get()));
    cudaFree(externalBuffer1);
    cudaFree(externalBuffer2);
    cusparseSpGEMM_destroyDescr(spgemm_descr);
}

template <ContextType device_type, typename value_type, typename index_type>
Sparse<device_type, value_type, index_type>::Sparse(Sparse<device_type, value_type, index_type>& mtx_in) : Mtx<device_type>(mtx_in.get_context())
{
//    nnz_ = mtx_in.get_nnz();
//    size_ = mtx_in.get_size();
//    values_ = mtx_in.get_values();
//    row_ptrs_ = mtx_in.get_row_ptrs();
//    col_idxs_ = mtx_in.get_col_idxs();
//    mtx_  = mtx_in.get_mtx();
//    descr_ = mtx_in.get_descriptor();
}


template class Sparse<CUDA, double, int>;
template class Sparse<CUDA, float, int>;
template class Sparse<CUDA, __half, int>;
template class Sparse<CPU, double, int>;
template class Sparse<CPU, float, int>;


template void Sparse<CUDA, double, magma_int_t>::copy_from(matrix::Sparse<CUDA, double, magma_int_t>* mtx);
template void Sparse<CUDA, double, magma_int_t>::copy_from(matrix::Sparse<CUDA, float, magma_int_t>* mtx);
template void Sparse<CUDA, double, magma_int_t>::copy_from(matrix::Sparse<CUDA, __half, magma_int_t>* mtx);
template void Sparse<CUDA, float, magma_int_t>::copy_from(matrix::Sparse<CUDA, float, magma_int_t>* mtx);
template void Sparse<CUDA, float, magma_int_t>::copy_from(matrix::Sparse<CUDA, double, magma_int_t>* mtx);
template void Sparse<CUDA, float, magma_int_t>::copy_from(matrix::Sparse<CUDA, __half, magma_int_t>* mtx);
template void Sparse<CPU, double, magma_int_t>::copy_from(matrix::Sparse<CPU, double, magma_int_t>* mtx);
template void Sparse<CPU, double, magma_int_t>::copy_from(matrix::Sparse<CPU, float, magma_int_t>* mtx);
template void Sparse<CPU, float, magma_int_t>::copy_from(matrix::Sparse<CPU, float, magma_int_t>* mtx);
template void Sparse<CPU, float, magma_int_t>::copy_from(matrix::Sparse<CPU, double, magma_int_t>* mtx);
template void Sparse<CUDA, __half, magma_int_t>::copy_from(matrix::Sparse<CUDA, float, magma_int_t>* mtx);
template void Sparse<CUDA, __half, magma_int_t>::copy_from(matrix::Sparse<CUDA, double, magma_int_t>* mtx);
template void Sparse<CUDA, __half, magma_int_t>::copy_from(matrix::Sparse<CUDA, __half, magma_int_t>* mtx);



}   // end of namespace matrix
}   // end of namespace rls
