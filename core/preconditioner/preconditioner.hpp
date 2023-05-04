#ifndef RLS_PRECONDITIONER_HPP
#define RLS_PRECONDITIONER_HPP


#include <iostream>
#include <memory>

#include "../dense/dense.hpp"
#include "../memory/magma_context.hpp"


namespace rls {
namespace preconditioner {


template<typename value_type_in, typename value_type, typename index_type, ContextType device_type>
class gaussian;

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

enum PrecondType {
    Undefined_PrecondType, 
    Gaussian,
    GENERALIZED_SPLIT
};

struct logger {
    magma_int_t runs_ = 1;
    magma_int_t warmup_runs_ = 0;
    double runtime_ = 0.0;
    double runtime_sketch_ = 0.0;
    double runtime_qr_ = 0.0;
};


template<ContextType device_type=CUDA>
class generic_preconditioner {
    protected:
        std::shared_ptr<Context<device_type>> context_;
        PrecondValueType type_selection_ = Undefined_PrecondValueType;
        PrecondType type_;
        logger logger_;

    public:
        generic_preconditioner() { }

        PrecondValueType get_precond_valuetype() { return type_selection_; }

        virtual void generate() {}

        virtual void compute() = 0;

        double get_runtime() {
            return logger_.runtime_;
        }
};


template <typename value_type, typename index_type, ContextType device_type=CUDA>
class preconditioner : public generic_preconditioner<device_type> {
public:
    preconditioner() {}

    virtual preconditioner* get() { return this; }

    virtual void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u) = 0;

    virtual value_type* get_values() { return precond_mtx_->get_values(); }

    virtual dim2 get_size() { return precond_mtx_->get_size(); }

    std::shared_ptr<matrix::Dense<value_type, device_type>> get_precond() { return precond_mtx_; }

    std::shared_ptr<matrix::Dense<value_type, device_type>> get_mtx() { return mtx_; }

protected:
    std::shared_ptr<matrix::Dense<value_type, device_type>> precond_mtx_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> mtx_;
    PrecondType precond_type_ = Undefined_PrecondType;
};


}   // namespace preconditioner
}   // namepsace rls


#endif
