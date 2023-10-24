#ifndef COUNTSKETCH_HPP
#define COUNTSKETCH_HPP


#include "../../include/base_types.hpp"
#include "../../cuda/sketch/gaussiansketch.cuh"
#include "../../cuda/sketch/countsketch.cuh"
#include "../../utils/convert.hpp"
#include "../../utils/io.hpp"
#include "../memory/magma_context.hpp"
#include "../matrix/dense/dense.hpp"
#include "../matrix/sparse/sparse.hpp"
#include "../blas/blas.hpp"
#include "sketch.hpp"


namespace rls {
namespace sketch{


//template void countsketch_impl(std::shared_ptr<matrix::Sparse<CUDA, double, magma_int_t>> sketch);
//template void countsketch_impl(std::shared_ptr<matrix::Sparse<CUDA, float, magma_int_t>> sketch);

//template void countsketch_impl(std::shared_ptr<matrix::Sparse<CUDA, double, magma_int_t>> sketch);
//template void countsketch_impl(std::shared_ptr<matrix::Sparse<CUDA, float, magma_int_t>> sketch);

//template void countsketch_impl(std::shared_ptr<matrix::Sparse<CUDA, double, magma_int_t>> sketch);
//template void countsketch_impl(std::shared_ptr<matrix::Sparse<CUDA, float, magma_int_t>> sketch);
//template void countsketch_impl(std::shared_ptr<matrix::Sparse<CUDA, __half, magma_int_t>> sketch);


}   // end of namespace sketch


//namespace utils {
//template <typename value_type_in, typename value_type_out, typename index_type>
//void convert(std::shared_ptr<Context<CUDA>> context, index_type num_rows, index_type num_cols, value_type_in* values_in, index_type ld_in,
//             value_type_out* values_out, index_type ld_out);
//}
template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
class CountSketch : public SketchOperator<device_type, value_type_apply, index_type> {

public:

    static std::unique_ptr<CountSketch<device_type, value_type, value_type_apply, index_type>>
            create(std::shared_ptr<Context<device_type>> context, size_t k, dim2 size);

    static std::unique_ptr<
        CountSketch<device_type, value_type, value_type_apply, index_type>>
            create(std::shared_ptr<Context<device_type>> context, size_t k,
                   std::string& filename_mtx);

    void apply(std::shared_ptr<MtxOp<device_type>> rhs,
               std::shared_ptr<matrix::Dense<device_type, value_type_apply>> result);

    value_type_apply* get_values();

    dim2 get_size();

    void set_mtx(std::shared_ptr<matrix::Sparse<device_type, value_type_apply, index_type>> t);

    std::shared_ptr<matrix::Sparse<device_type, value_type_apply, index_type>> get_mtx();

    CountSketch() {}

    CountSketch(CountSketch<device_type, value_type, value_type_apply, index_type> &t);

    CountSketch<device_type, value_type, value_type_apply, index_type>& operator=(CountSketch<device_type, value_type, value_type_apply, index_type>& t);

    size_t get_nnz_per_col();

private:

    size_t nnz_per_col_ = 0;
    dim2 size_;
    std::shared_ptr<matrix::Sparse<device_type, value_type_apply, index_type>> mtx_;

    void convert_to(
        int k,
        matrix::Sparse<device_type, value_type, index_type>* mtx_in,
        matrix::Sparse<device_type, value_type_apply, index_type>* mtx_out);

    CountSketch(std::shared_ptr<Context<device_type>> context, size_t k, dim2 size);

    CountSketch(std::shared_ptr<Context<device_type>> context, size_t k, std::string& filename_mtx);
};


}   // end of namespace rls


#endif
