#include "../../core/matrix/dense/dense.hpp"
#include "../../core/memory/magma_context.hpp"


namespace rls {
namespace sketch{


void gaussian_sketch_impl(std::shared_ptr<matrix::Dense<CUDA, double>> mtx);

void gaussian_sketch_impl(std::shared_ptr<matrix::Dense<CUDA, float>> mtx);


}   // end of namespace sketch
}   // end of namespace rls
