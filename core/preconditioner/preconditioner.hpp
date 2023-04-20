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


template<ContextType device_type=CUDA>
class generic_preconditioner {
    protected:
        std::shared_ptr<Context<device_type>> context_;
        PrecondValueType type_selection_ = Undefined_PrecondValueType;
        PrecondType type_;

    public:
        generic_preconditioner() { }

        PrecondValueType get_precond_valuetype() { return type_selection_; }

        virtual void generate() {}

        virtual void compute() = 0;
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

    template<typename value_type_out>
    std::shared<preconditioner<value_type_out, index_type, device_type>> convert_to();

    // template<typename value_type_in>
    // void convert_from(preconditioner<value_type_in, index_type, device_type>* precond) {
    //     std::cout << "in convert\n";
    //     auto c = precond->get_mtx()->get_context();
    //     std::cout << "before create mtx\n";
    //     // this->mtx_ = matrix::Dense<value_type, device_type>::create(c);
    //     // std::cout << "after create mtx\n";
    //     // this->precond_mtx_ = matrix::Dense<value_type, device_type>::create(c);
    //     // = std::make_shared<matrix::Dense<value_type, device_type>>(matrix::Dense<value_type, device_type>::create());
    //         // std::cout << "test\n";
    //     // this->precond_mtx_ = std::make_shared<matrix::Dense<value_type, device_type>>(c);
    //     // this->precond_mtx_ = matrix::Dense<value_type, device_type>(precond->get_mtx()->get_context());
    //     auto s = precond->get_mtx();
    //     std::cout << "before copy from\n";
    //     std::cout << "before print_test()\n";
    //     this->mtx_->print_test();
    //     std::cout << "after print_test()\n";
    //     this->mtx_->copy_from(precond->get_mtx());
    //     std::cout << "after convert\n";
    // }

protected:
    std::shared_ptr<matrix::Dense<value_type, device_type>> precond_mtx_;
    std::shared_ptr<matrix::Dense<value_type, device_type>> mtx_;
    PrecondType precond_type_ = Undefined_PrecondType;
};


}   // namespace preconditioner
}   // namepsace rls


#endif
