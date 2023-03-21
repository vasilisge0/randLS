#ifndef ABSTRACT_PRECONDITIONER_HPP
#define ABSTRACT_PRECONDITIONER_HPP

#include "../dense/dense.hpp"
#include <memory>
#include "../memory/magma_context.hpp"
// #include "gaussian.hpp"

#include <iostream>



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
        std::shared_ptr<MagmaContext> context;
        PrecondValueType type_selection = Undefined_PrecondValueType;
        PrecondType type;

    public:
        generic_preconditioner() {}

        PrecondValueType get_precond_valuetype() { return type_selection; }

    virtual void generate();

    void test();

};


template <typename value_type_in, typename value_type, typename index_type>
class preconditioner : public generic_preconditioner {
public:
    preconditioner(){}

    virtual preconditioner* get() { return this; }

    virtual void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u) {}

    virtual void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u, magma_queue_t queue) {}

    virtual value_type* get_values() {}

    virtual dim2 get_size() {}

    std::shared_ptr<matrix::dense<value_type>> get_mtx() {
        return mtx;
    }

    void generate(matrix::dense<value_type>* mtx) { }

    void set_type() {
        if ((typeid(value_type_in) == typeid(double)) && (typeid(value_type_in) == typeid(double)))
        {
            type_selection = FP64_FP64;
        }
        else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(double)))
        {
            type_selection = FP32_FP64;
        }
        else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(double)))
        {
            type_selection = FP16_FP64;
        }
        else if ((typeid(value_type_in) == typeid(float)) && (typeid(value_type_in) == typeid(float)))
        {
            type_selection = FP32_FP32;
        }
        else if ((typeid(value_type_in) == typeid(__half)) && (typeid(value_type_in) == typeid(float)))
        {
            type_selection = FP16_FP32;
        }
    }

protected:
    std::shared_ptr<matrix::dense<value_type>> mtx;
    PrecondType precond_type = Undefined_PrecondType;
};


}
}


#endif
