#ifndef RLS_PRECONDITIONER_HPP
#define RLS_PRECONDITIONER_HPP


#include <iostream>
#include <memory>

#include "../dense/dense.hpp"
#include "../memory/magma_context.hpp"


namespace rls {
namespace preconditioner {


template<typename value_type_in, typename value_type, typename index_type>
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
    Gaussian
};


class generic_preconditioner {
    protected:
        std::shared_ptr<Context> context_;
        PrecondValueType type_selection_ = Undefined_PrecondValueType;
        PrecondType type_;

    public:
        generic_preconditioner() { }

        PrecondValueType get_precond_valuetype() { return type_selection_; }

        virtual void generate();
};


template <typename value_type_in, typename value_type, typename index_type>
class preconditioner : public generic_preconditioner {
public:
    preconditioner() {}

    virtual preconditioner* get() { return this; }

    virtual void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u) = 0;

    virtual void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u, magma_queue_t queue) = 0;

    virtual value_type* get_values() { return mtx_->get_values(); }

    virtual dim2 get_size() { return mtx_->get_size(); }

    std::shared_ptr<matrix::dense<value_type>> get_mtx() { return mtx_; }

    virtual void generate(matrix::dense<value_type>* mtx_in) = 0;

    virtual void compute(matrix::dense<value_type>* mtx) = 0;

    // virtual void compute() = 0;

    void set_type() {
        if ((typeid(value_type_in) == typeid(double)) && (typeid(value_type_in) == typeid(double)))
        {
            type_selection_ = FP64_FP64;
        }
        else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(double)))
        {
            type_selection_ = FP32_FP64;
        }
        else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(double)))
        {
            type_selection_ = FP16_FP64;
        }
        else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(float)))
        {
            type_selection_ = FP32_FP32;
        }
        else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(float)))
        {
            type_selection_ = FP16_FP32;
        }
    }

protected:
    std::shared_ptr<matrix::dense<value_type>> mtx_;
    PrecondType precond_type_ = Undefined_PrecondType;
};


}   // namespace preconditioner
}   // namepsace rls


#endif
