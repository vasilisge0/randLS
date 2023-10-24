#ifndef RLS_DENSE_HPP
#define RLS_DENSE_HPP


#include <iostream>
#include <string>


//#include "../../memory/magma_context.hpp"
//#include "../../memory/memory.hpp"
//#include "../../../include/base_types.hpp"
//#include "../../../utils/io.hpp"
#include "../../../utils/convert.hpp"
#include "../mtxop.hpp"
#include <ginkgo/ginkgo.hpp>


namespace rls {
namespace matrix {


template <ContextType device_type, typename value_type>
class Dense : public MtxOp<device_type> {

private:

    void malloc();

    void free();

    Dense(std::shared_ptr<Context<device_type>> context);

    Dense(std::shared_ptr<Context<device_type>> context, dim2 size);

    Dense(std::shared_ptr<Context<device_type>> context,
        std::string& filename_mtx);

    Dense(std::shared_ptr<Context<device_type>> context, dim2 size, value_type* values);

    Dense(std::shared_ptr<Context<device_type>> context, dim2 size, magma_int_t ld, value_type* values);

public:

    void apply(Dense<device_type, value_type>* rhs, Dense<device_type, value_type>* result);

    std::unique_ptr<Dense<device_type, value_type>> transpose();

    std::unique_ptr<Dense<device_type, value_type>> row_to_col_order();

    gko::LinOp* get_mtx();

    std::shared_ptr<gko::LinOp> get_mtx_shared();

    value_type* get_values() { return values_; }

    magma_int_t get_ld() { return ld_; }

    void set_matrix(std::shared_ptr<gko::LinOp> mtx_in);

    static std::unique_ptr<Dense<device_type, value_type>> create(
        std::shared_ptr<Context<device_type>> context);

    static std::unique_ptr<Dense<device_type, value_type>> create(
        std::shared_ptr<Context<device_type>> context, dim2 size);

    static std::unique_ptr<Dense<device_type, value_type>> create(
        std::shared_ptr<Context<device_type>> context, std::string& filename_mtx);

    static std::unique_ptr<Dense<device_type, value_type>> create(
        std::shared_ptr<Context<device_type>> context, dim2 size, value_type* values);

    static std::unique_ptr<Dense<device_type, value_type>> create(
        std::shared_ptr<Context<device_type>> context, dim2 size, magma_int_t ld, value_type* values);

    static std::unique_ptr<Dense<device_type, value_type>> create_submatrix(
        matrix::Dense<device_type, value_type>* mtx_in, span cspan);

    static std::unique_ptr<Dense<device_type, value_type>> create_submatrix(
        matrix::Dense<device_type, value_type>* mtx_in, span rspan, span cspan);

    static std::unique_ptr<Dense<device_type, value_type>> create_subcol(
        matrix::Dense<device_type, value_type>* mtx_in, span rspan, int col);

    template<typename value_type_in>
    void copy_from(matrix::Dense<device_type, value_type_in>* mtx);

    size_t get_num_elems();

    magma_int_t get_alloc_elems();

    void zeros();

    void eye();

    const value_type* get_const_values() const;

    dim2 get_size();

    cusparseDnMatDescr_t get_descriptor();

    ~Dense();

private:

    Dense(Dense&& mtx);

    Dense<device_type, value_type>& operator=(Dense<device_type, value_type>&& mtx);

    dim2 size_ = {0, 0};
    magma_int_t ld_ = 0;
    magma_int_t alloc_elems = 0;
    value_type* values_ = nullptr;
    std::shared_ptr<gko::LinOp> mtx;
    cusparseDnMatDescr_t descr_;
    bool wrapper_ = false;
};


}  // end of namespace matrix
}  // end of namespace rls


#endif
