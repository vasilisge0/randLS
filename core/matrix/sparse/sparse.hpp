#ifndef RLS_SPARSE_HPP
#define RLS_SPARSE_HPP


#include <iostream>
#include <string>


#include <ginkgo/ginkgo.hpp>
#include "../../memory/magma_context.hpp"
#include "../../memory/memory.hpp"
#include "../../../include/base_types.hpp"
#include "../../../utils/io.hpp"
#include "../../../utils/convert.hpp"
#include "../mtxop.hpp"


namespace rls {
namespace matrix {


template <ContextType device_type, typename value_type, typename index_type>
class Sparse : public MtxOp<device_type> {
public:

    static std::unique_ptr<Sparse> create(std::shared_ptr<Context<device_type>> context, dim2 size);

    static std::unique_ptr<Sparse> create(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz);

    static std::unique_ptr<Sparse> create(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz,
        value_type* values, index_type* row_ptrs, index_type* col_idxs);

    static std::unique_ptr<Sparse> create(std::shared_ptr<Context<device_type>> context,
        std::string& filename_mtx);

    static std::unique_ptr<Sparse> create(std::shared_ptr<Context<device_type>> context,
        std::shared_ptr<gko::LinOp> mtx);

    std::shared_ptr<gko::LinOp> get_mtx();

    gko::LinOp* get_raw_mtx();

    std::shared_ptr<Context<device_type>> get_context();

    dim2 get_size();

    size_t get_nnz();

    value_type* get_values();

    index_type* get_row_ptrs();

    index_type* get_col_idxs();

    template<typename value_type_in>
    void copy_from(matrix::Sparse<device_type, value_type_in, index_type>* mtx);

    size_t get_num_elems();

    std::unique_ptr<Sparse> transpose();

    Sparse(Sparse<device_type, value_type, index_type>&& t);

    void apply(Sparse<device_type, value_type, index_type>* rhs, Sparse<device_type, value_type, index_type>* result);

    void apply(value_type alpha, Dense<device_type, value_type>* rhs, value_type beta, Dense<device_type, value_type>* result);

    void to_dense(Dense<device_type, value_type>* result);

    void set_row_ptrs(index_type* row_ptrs);

    void set_col_idxs(index_type* col_idxs);

    void set_values(value_type* values);

    void set_descriptor(cusparseSpMatDescr_t descr);

    cusparseSpMatDescr_t get_descriptor();

    ~Sparse();

    Sparse<device_type, value_type, index_type>& operator=(Sparse<device_type, value_type, index_type>& mtx_in);

    Sparse<device_type, value_type, index_type>& operator=(Sparse<device_type, value_type, index_type>&& mtx_in);

    Sparse(Sparse<device_type, value_type, index_type>& mtx_in);

private:

    Sparse(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz);

    Sparse(std::shared_ptr<Context<device_type>> context, dim2 size, size_t nnz,
        value_type* values, index_type* row_ptrs, index_type* col_idxs);

    Sparse(std::shared_ptr<Context<device_type>> context,
        std::string& filename_mtx);

    Sparse(std::shared_ptr<Context<device_type>> context, std::shared_ptr<gko::LinOp> mtx);

    Sparse(std::shared_ptr<Context<device_type>> context, dim2 size);

    size_t nnz_;
    dim2 size_;
    value_type* values_;
    index_type* row_ptrs_;
    index_type* col_idxs_;
    std::shared_ptr<gko::LinOp> mtx_;
    cusparseSpMatDescr_t descr_;
};

//template class Sparse<CUDA, double, magma_int_t>;
//template class Sparse<CUDA, float, magma_int_t>;
//template class Sparse<CUDA, __half, magma_int_t>; // To be added eventually.


} // end of namespace matrix
} // end of namespace rls


#endif
