#ifndef SKETCH_HPP
#define SKETCH_HPP


#include "../memory/magma_context.hpp"
#include "../matrix/dense/dense.hpp"
#include "../blas/blas.hpp"
#include "../../include/base_types.hpp"
#include "../../cuda/sketch/gaussiansketch.cuh"
#include "../../utils/convert.hpp"
#include "../../utils/io.hpp"


namespace rls {


class Sketch {};

template<ContextType device_type, typename value_type, typename index_type>
class SketchOperator : Sketch {
public:
    virtual void apply(std::shared_ptr<MtxOp<device_type>> rhs, std::shared_ptr<matrix::Dense<device_type, value_type>> result) = 0;

    virtual value_type* get_values() = 0;

    virtual dim2 get_size() = 0;

protected:
    std::shared_ptr<Context<device_type>> context_;
};



} // end of namespace rls

#endif
