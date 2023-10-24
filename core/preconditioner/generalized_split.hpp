#ifndef RLS_GENERALIZED_SPLIT_HPP
#define RLS_GENERALIZED_SPLIT_HPP


#include <iostream>
#include <memory>
#include "magma_v2.h"


#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../cuda/solver/lsqr_kernels.cuh"
#include "../../utils/io.hpp"
#include "../blas/blas.hpp"
#include "../matrix/dense/dense.hpp"
#include "../matrix/sparse/sparse.hpp"
#include "../matrix/mtxop.hpp"
#include "../memory/magma_context.hpp"
#include "preconditioner.hpp"
#include "sketchqr.hpp"


namespace rls {
namespace preconditioner {
namespace generalized_split {


template <ContextType device_type, typename value_type,
          typename value_type_internal, typename index_type>
int sketch_qr_impl(
    std::shared_ptr<Context<device_type>> context,
    std::shared_ptr<
        SketchOperator<device_type, value_type_internal, index_type>>
        sketch,
    std::shared_ptr<matrix::Dense<device_type, value_type>> mtx,
    std::shared_ptr<matrix::Dense<device_type, value_type>> precond_mtx,
    state<device_type, value_type, value_type_internal, index_type>* state);

template <typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
class Config : public preconditioner::Config {
public:
    Config() {}

    Config(double sampling_coefficient);

    static std::unique_ptr<Config<value_type, value_type_internal_0, value_type_precond_0, index_type>>
        create(double sampling_coefficient);

};

template <typename value_type, typename value_type_internal_0,
          typename index_type>
struct state {
    value_type_internal_0* dmtx_rp    = nullptr;
    value_type_internal_0* dsketch_rp = nullptr;
    value_type_internal_0* dresult_rp = nullptr;
    value_type* tau                   = nullptr;

    void allocate(index_type ld_mtx, index_type num_cols_mtx,
                  index_type num_rows_sketch, index_type num_cols_sketch,
                  index_type ld_sketch, index_type ld_r_factor);

    void free();
};


}  // end of namespace generalized_split


template <ContextType device_type, typename value_type, typename value_type_internal_0, typename value_type_precond_0, typename index_type>
class GeneralizedSplit
    : public PrecondOperator<device_type, value_type, index_type> {

public:

    void apply(std::shared_ptr<Context<device_type>> context, magma_trans_t trans,
        matrix::Dense<device_type, value_type>* u_vector);

    void generate();

    matrix::Dense<device_type, value_type>* get_precond();

    template<typename value_type_out>
    std::shared_ptr<GeneralizedSplit
        <device_type, value_type_out, value_type_out, value_type_out, index_type>> convert_to();

    static std::unique_ptr<
        GeneralizedSplit<device_type, value_type, value_type_internal_0, value_type_precond_0, index_type>>
    create(std::shared_ptr<MtxOp<device_type>> mtx,
           std::shared_ptr<SketchOperator<device_type, value_type_internal_0, index_type>> sketch,
           std::shared_ptr<preconditioner::Config> config);

private:

    GeneralizedSplit(
           std::shared_ptr<MtxOp<device_type>> mtx,
           std::shared_ptr<SketchOperator<device_type, value_type_internal_0, index_type>> sketch,
           std::shared_ptr<generalized_split::Config<value_type, value_type_internal_0, value_type_precond_0, index_type>> config);

    std::shared_ptr<generalized_split::Config<value_type, value_type_internal_0, value_type_precond_0, index_type>> config_;
    std::shared_ptr<SketchOperator<device_type, value_type_internal_0, index_type>> sketch_;
    std::unique_ptr<matrix::Dense<device_type, value_type_precond_0>> temp1;
    std::unique_ptr<matrix::Dense<device_type, value_type_precond_0>> temp2;
    std::unique_ptr<matrix::Dense<device_type, value_type_precond_0>> precond_mtx_internal_;
    std::shared_ptr<matrix::Dense<device_type, value_type>> precond_mtx_;
};  // end of class GeneralizedSplit


//template class GeneralizedSplit<CUDA, double, double, double, magma_int_t>;


}  // end namespace preconditioner
}  // end namespace rls


#endif
