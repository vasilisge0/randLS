#include <iostream>
#include "solver.hpp"
#include "lsqr.hpp"


namespace rls {
namespace solver {


// void solver::generate(std::string& filename_mtx, std::string& filename_rhs)
// {
//     // if (type_ == Lsqr) {
//         // switch(combined_value_type_)
//         // {
//             // case FP64_FP64:
//             // {
//                 // auto this_solver = dynamic_cast<lsqr<double, double, magma_int_t>*>(this);
//                 // this_solver->generate(filename_mtx, filename_rhs);
//                 // break;
//             // }
//             // case FP32_FP64:
//             // {
//                 // auto this_solver = dynamic_cast<lsqr<float, double, magma_int_t>*>(this);
//                 // this_solver->generate(filename_mtx, filename_rhs);
//                 // break;
//             // }
//             // case TF32_FP64:
//             // {
//                 // auto this_solver = dynamic_cast<lsqr<float, double, magma_int_t>*>(this);
//                 // this_solver->generate(filename_mtx, filename_rhs);
//                 // break;
//             // }
//             // case FP16_FP64:
//             // {
//                 // auto this_solver = dynamic_cast<lsqr<__half, double, magma_int_t>*>(this);
//                 // this_solver->generate(filename_mtx, filename_rhs);
//                 // break;
//             // }
//             // case FP32_FP32:
//             // {
//                 // auto this_solver = dynamic_cast<lsqr<float, float, magma_int_t>*>(this);
//                 // this_solver->generate(filename_mtx, filename_rhs);
//                 // break;
//             // }
//             // case FP16_FP32:
//             // {
//                 // auto this_solver = dynamic_cast<lsqr<__half, float, magma_int_t>*>(this);
//                 // this_solver->generate(filename_mtx, filename_rhs);
//                 // break;
//             // }
//             // case TF32_FP32:
//             // {
//                 // auto this_solver = dynamic_cast<lsqr<float, float, magma_int_t>*>(this);
//                 // this_solver->generate(filename_mtx, filename_rhs);
//                 // break;
//             // }
//             // default:
//                 // break;
//         // }
//     // }    
// }

// void solver::run()
// {
//     // if (type_ == Lsqr) {
//     //     switch(combined_value_type_)
//     //     {
//     //         case FP64_FP64:
//     //         {
//     //             auto this_solver = dynamic_cast<lsqr<double, double, magma_int_t>*>(this);
//     //             this_solver->run();
//     //             break;
//     //         }
//     //         case FP32_FP64:
//     //         {
//     //             auto this_solver = dynamic_cast<lsqr<float, double, magma_int_t>*>(this);
//     //             this_solver->run();
//     //             break;
//     //         }
//     //         case TF32_FP64:
//     //         {
//     //             auto this_solver = dynamic_cast<lsqr<float, double, magma_int_t>*>(this);
//     //             this_solver->run();
//     //             break;
//     //         }
//     //         case FP16_FP64:
//     //         {
//     //             auto this_solver = dynamic_cast<lsqr<__half, double, magma_int_t>*>(this);
//     //             this_solver->run();
//     //             break;
//     //         }
//     //         case FP32_FP32:
//     //         {
//     //             auto this_solver = dynamic_cast<lsqr<float, float, magma_int_t>*>(this);
//     //             this_solver->run();
//     //             break;
//     //         }
//     //         case FP16_FP32:
//     //         {
//     //             auto this_solver = dynamic_cast<lsqr<__half, float, magma_int_t>*>(this);
//     //             this_solver->run();
//     //             break;
//     //         }
//     //         case TF32_FP32:
//     //         {
//     //             auto this_solver = dynamic_cast<lsqr<float, float, magma_int_t>*>(this);
//     //             this_solver->run();
//     //             break;
//     //         }
//     //         default:
//     //             break;
//     //     }
//     // }
// }


}
}
