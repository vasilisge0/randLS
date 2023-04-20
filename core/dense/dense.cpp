#include <iostream>
#include <string>


#include "dense.hpp"
#include "../../cuda/solver/lsqr_kernels.cuh"
#include "../../include/base_types.hpp"
#include "../../utils/io.hpp"
#include "../../utils/convert.hpp"
#include "../memory/memory.hpp"


namespace rls {
namespace matrix {


// template<typename value_type, ContextType device_type>
// template<typename value_type_in>
// void Dense<value_type, device_type>::copy_from(std::shared_ptr<matrix::Dense<value_type_in, device_type>> mtx) {

//     this->size_ = mtx->get_size();
//     this->ld_ = mtx->get_ld();
//     this->context_ = mtx->get_context();

//     if (this->alloc_elems == 0) {
//         this->malloc();
//     }

//     // utils::convert(this->size_[0], this->size_[1], mtx->get_values(), mtx->get_ld(),
//         // this->values_, this->ld_);
// }



// // template<typename value_type, ContextType device_type>
// template
// void Dense<double, CUDA>::copy_from(std::shared_ptr<matrix::Dense<float, CUDA>> mtx);

}  // namespace matrix
}  // namespace rls