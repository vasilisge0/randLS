#ifndef RLS_PRECONDITIONER_HPP
#define RLS_PRECONDITIONER_HPP


#include <iostream>
#include <memory>


#include "../matrix/dense/dense.hpp"
#include "../matrix/sparse/sparse.hpp"
#include "../matrix/mtxop.hpp"
#include "../memory/magma_context.hpp"


namespace rls {
namespace preconditioner {


enum MemoryStrategy { Free_Internal_Precision, Keep_Internal_Precision };

class PrecondConfig {
    MemoryStrategy compute_memory_strategy_ = Keep_Internal_Precision;
};

class Config {
protected:
    double sampling_coefficient_ = 1.0;
    MemoryStrategy mem_strategy_ = Keep_Internal_Precision;

public:
    Config() {}

    Config(double sampling_coefficient);

    double get_sampling_coefficient();

    void set_sampling_coefficient(double sampling_coefficient);

    void set_memory_strategy(MemoryStrategy mem_strategy);

};

template <ContextType device_type, typename value_type, typename value_type_in,
          typename value_type_precond, typename index_type>
class SketchQr;

enum PrecondValueType {
    Undefined_PrecondValueType,
    FP64_FP64,
    FP32_FP64,
    TF32_FP64,
    FP16_FP64,
    FP32_FP32,
    TF32_FP32,
    FP16_FP32
};

class logger {
public:
    magma_int_t runs_        = 1;
    magma_int_t warmup_runs_ = 0;
    double runtime_          = 0.0;
    double runtime_sketch_   = 0.0;
    double runtime_qr_       = 0.0;
};


}  // end of namespace preconditioner


class Preconditioner {
public:
    virtual void generate() = 0;

    preconditioner::logger get_logger();

private:
    preconditioner::logger logger_;
};

template <ContextType device_type, typename value_type_apply,
          typename index_type>
class PrecondOperator : public Preconditioner {
public:

    PrecondOperator(std::shared_ptr<MtxOp<device_type>> mtx);

    virtual void apply(std::shared_ptr<Context<device_type>> context, magma_trans_t trans,
        rls::matrix::Dense<device_type, value_type_apply>* u_vector_part_1) = 0;

    std::shared_ptr<MtxOp<device_type>> get_mtx();

    std::shared_ptr<Context<device_type>> get_context();

private:
    std::shared_ptr<Context<device_type>> context_;
    std::shared_ptr<MtxOp<device_type>> mtx_;
};


}  // end of namespace rls


#endif
