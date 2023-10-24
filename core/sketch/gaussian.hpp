#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP


#include <iostream>
#include "sketch.hpp"
#include "../memory/magma_context.hpp"
#include "../matrix/dense/dense.hpp"
#include "../blas/blas.hpp"
#include "../../include/base_types.hpp"
#include "../../cuda/sketch/gaussiansketch.cuh"
#include "../../utils/convert.hpp"
#include "../../utils/io.hpp"


namespace rls {


template <ContextType device_type, typename value_type, typename index_type>
class GaussianSketchConfig {
    double sampling_coefficient;
};

template<ContextType device_type, typename value_type, typename value_type_apply, typename index_type>
class GaussianSketch : public SketchOperator<device_type, value_type_apply, index_type> {
public:

    static std::unique_ptr<GaussianSketch<device_type, value_type, value_type_apply, index_type>> create(std::shared_ptr<Context<device_type>> context, dim2 size);

    static std::unique_ptr<GaussianSketch<device_type, value_type, value_type_apply, index_type>> create(
        std::shared_ptr<Context<device_type>> context, std::string& filename_mtx);

    void apply(std::shared_ptr<MtxOp<device_type>> rhs, std::shared_ptr<matrix::Dense<device_type, value_type_apply>> result);

    dim2 get_size();

    value_type_apply* get_values();

    std::shared_ptr<matrix::Dense<device_type, value_type_apply>> get_mtx();

private:

    std::shared_ptr<matrix::Dense<device_type, value_type_apply>> mtx_;

    void convert_to(matrix::Dense<device_type, value_type>* mtx_in,
                    matrix::Dense<device_type, value_type_apply>* mtx_out);

    GaussianSketch(std::shared_ptr<Context<device_type>> context, dim2 size);

    GaussianSketch(std::shared_ptr<Context<device_type>> context,
        std::string& filename_mtx);
};

//template<>
//class GaussianSketch<CUDA, double, double, magma_int_t>;


}   // end of namespace rls


#endif
