#include "preconditioner.hpp"
#include "gaussian.hpp"

namespace rls {
namespace preconditioner {

void generic_preconditioner::generate() {
    if (this->type == Gaussian) {
        switch(type_selection)
        {
            case FP64_FP64:
            {
                auto this_precond = dynamic_cast<gaussian<double, double, magma_int_t>*>(this);
                break;
            }
            case FP32_FP64:
            {
                auto this_solver = dynamic_cast<gaussian<float, double, magma_int_t>*>(this);
                break;
            }
            case TF32_FP64:
            {
                auto this_solver = dynamic_cast<gaussian<float, double, magma_int_t>*>(this);
                break;
            }
            case FP16_FP64:
            {
                auto this_solver = dynamic_cast<gaussian<__half, double, magma_int_t>*>(this);
                break;
            }
            case FP32_FP32:
            {
                auto this_solver = dynamic_cast<gaussian<float, float, magma_int_t>*>(this);
                break;
            }
            case FP16_FP32:
            {
                auto this_solver = dynamic_cast<gaussian<__half, float, magma_int_t>*>(this);
                break;
            }
            case TF32_FP32:
            {
                auto this_solver = dynamic_cast<gaussian<float, float, magma_int_t>*>(this);
                break;
            }
            default:
                break;
        }
    }

}

// template <typename value_type_in, typename value_type, typename index_type>
// void preconditioner<value_type_in, value_type, index_type>::generate(matrix::dense<value_type>* mtx) {
    // if (this->type == Gaussian) {
        // auto this_precond = dynamic_cast<gaussian<value_type_in, value_type, index_type>*>(this);
        // this_precond->generate(mtx);
    // }
// }
// 
// template <typename value_type_in, typename value_type, typename index_type>
// void preconditioner<value_type_in, value_type, index_type>::test() {
// 
// }

void generic_preconditioner::test() {
    std::cout << "testing\n";
}

// template <typename value_type_in, typename value_type, typename index_type>
// void preconditioner<value_type_in, value_type, index_type>::generate(matrix::dense<value_type>* mtx) {
//     if (this->type == Gaussian) {
//         auto this_precond = dynamic_cast<gaussian<value_type_in, value_type, index_type>*>(this);
//         this_precond->generate(mtx);
//     }
// }

// template <typename value_type_in, typename value_type, typename index_type>
// void preconditioner<value_type_in, value_type, index_type>::generate(matrix::dense<value_type>* mtx) {
    // if (this->type == Gaussian) {
        // auto this_precond = dynamic_cast<gaussian<value_type_in, value_type, index_type>*>(this);
        // this_precond->generate(mtx);
    // }
// }

}
}
