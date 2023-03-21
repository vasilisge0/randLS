#ifndef ABSTRACT_PRECONDITIONER_HPP
#define ABSTRACT_PRECONDITIONER_HPP

#include "../dense/dense.hpp"
#include <memory>
#include "../memory/detail.hpp"
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
        detail::magma_info info;
        // magma_queue_t queue;
        PrecondValueType type_selection = Undefined_PrecondValueType;
        PrecondType type;

    public:
        generic_preconditioner() {
        }

        PrecondValueType get_precond_valuetype() {
            return type_selection;
        }

    virtual void generate();

    void test();

};


template <typename value_type_in, typename value_type, typename index_type>
class preconditioner : public generic_preconditioner {
public:
    preconditioner(){}

    virtual preconditioner* get() { return this; }

    virtual void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u) {

    }

    std::shared_ptr<matrix::dense<value_type>> get_mtx() {
        return mtx;
    }

    // void generate(matrix::dense<value_type>* mtx);
    void generate(matrix::dense<value_type>* mtx) {
        std::cout << "GENERATE PRECOND\n";
        // if (this->type == Gaussian) {
            // auto this_precond = dynamic_cast<gaussian<value_type_in, value_type, index_type>*>(this);
            // auto this_precond = dynamic_cast<gaussian<double, double, magma_int_t>*>(precond);
            // this_precond->generate(mtx);
        // }
    }


    void set_type() {
        std::cout << "TYPE in preconditioner set_type: !!!!!!!! " << '\n';
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

    // virtual void generate(matrix::dense<value_type>* mtx);

protected:
    std::shared_ptr<matrix::dense<value_type>> mtx;
    PrecondType precond_type = Undefined_PrecondType;
};


}
}


#endif
