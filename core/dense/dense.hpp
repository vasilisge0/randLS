#ifndef RLS_DENSE_HPP
#define RLS_DENSE_HPP


#include <iostream>
#include <string>


#include "../../cuda/solver/lsqr_kernels.cuh"
#include "../../include/base_types.hpp"
#include "../../utils/io.hpp"
#include "../memory/memory.hpp"


namespace rls {
namespace matrix {



template <typename value_type, ContextType device_type>
class Dense {
public:

    Dense(std::shared_ptr<Context<device_type>> context) {
        this->context_ = context;
    }

    Dense(std::shared_ptr<Context<device_type>> context, dim2 size) {
        this->context_ = context;
        this->size_ = size;
        malloc();
    }

    Dense(std::shared_ptr<Context<device_type>> context, std::string& filename_mtx) {
        this->context_ = context;
        io::read_mtx_size((char*)filename_mtx.c_str(), &this->size_[0], &this->size_[1]);
        malloc();
        auto t = context->get_queue();
        io::read_mtx_values<value_type, device_type>(this->context_, (char*)filename_mtx.c_str(),
          this->size_, this->values_);
    }

    Dense(const Dense&& mtx) {
        this->device_ = mtx.device_;
        this->size_ = mtx.size_;
        this->values_ = mtx.values_;
        mtx.values = nullptr;
        this->context_ = mtx.context_;
    }

    void zeros()
    {
        memory::zeros<value_type, device_type>(this->size_, this->values_);
    }

    static std::unique_ptr<Dense<value_type, device_type>> create(std::shared_ptr<Context<device_type>>
        context) {
        auto tmp = new Dense<value_type, device_type>(context);
        return std::unique_ptr<Dense<value_type, device_type>>(tmp);
    }

    static std::unique_ptr<Dense<value_type, device_type>> create(
        std::shared_ptr<Context<device_type>> context, dim2 size) {
        auto tmp = new Dense<value_type, device_type>(context, size);
        return std::unique_ptr<Dense<value_type, device_type>>(tmp);
    }

    static std::unique_ptr<Dense<value_type, device_type>> create(
        std::shared_ptr<Context<device_type>> context, std::string& filename_mtx)
    {
        auto tmp = new Dense<value_type, device_type>(context, filename_mtx);
        auto queue = context->get_queue();
        return std::unique_ptr<Dense<value_type, device_type>>(tmp);
    }

    value_type* get_values() { return this->values_; }

    const value_type* get_const_values() const { return this->values_; }

    dim2& get_size() { return this->size_; }

    ~Dense() { free(); }

    std::shared_ptr<Context<device_type>> get_context() { return this->context_; }

private:

    void malloc() {
        switch (device_type) {
        case CPU:
            memory::malloc_cpu(&this->values_, this->size_[0] * this->size_[1]);
            break;
        case CUDA:
            memory::malloc(&this->values_, this->size_[0] * this->size_[1]);
            break;
        default:
            break;
        }
    }

    void free() {
        switch (device_type) {
        case CPU:
            memory::free_cpu(this->values_);
            break;
        case CUDA:
            memory::free(this->values_);
            break;
        default:
            break;
        }
    }

    std::shared_ptr<Context<device_type>> context_;
    dim2 size_;
    value_type* values_ = nullptr;
    ContextType device_;
};


}  // namespace matrix
}  // namespace rls

#endif
