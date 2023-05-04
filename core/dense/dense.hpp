#ifndef RLS_DENSE_HPP
#define RLS_DENSE_HPP


#include <iostream>
#include <string>


// #include "../../cuda/solver/lsqr_kernels.cuh"

#include "../memory/magma_context.hpp"
#include "../../include/base_types.hpp"
#include "../../utils/io.hpp"
#include "../../utils/convert.hpp"
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
        this->ld_ = size[0];
        malloc();
        this->alloc_elems = size[0] * size[1];
    }

    Dense(std::shared_ptr<Context<device_type>> context, std::string& filename_mtx) {
        this->context_ = context;
        io::read_mtx_size((char*)filename_mtx.c_str(), &this->size_[0], &this->size_[1]);
        this->ld_ = this->size_[0];
        malloc();
        this->alloc_elems = size_[0] * size_[1];
        auto t = context->get_queue();
        io::read_mtx_values<value_type, device_type>(this->context_, (char*)filename_mtx.c_str(),
            this->size_, this->values_);
    }

    Dense(const Dense&& mtx) {
        this->device_ = mtx.device_;
        this->size_ = mtx.size_;
        this->values_ = mtx.values_;
        this->ld_ = mtx.ld;
        mtx.values = nullptr;
        this->context_ = mtx.context_;
    }

    template<typename value_type_in>
    void copy_from(std::shared_ptr<matrix::Dense<value_type_in, device_type>> mtx)  {

        this->size_ = mtx->get_size();
        this->ld_ = mtx->get_ld();
        this->context_ = mtx->get_context();

        if (this->alloc_elems == 0) {
            this->malloc();
        }

        utils::convert(this->context_, this->size_[0], this->size_[1],
            mtx->get_values(), mtx->get_ld(), this->values_, this->ld_);
    }

    Dense<value_type, device_type>& operator=(Dense<value_type, device_type>&& mtx) {
        if (*this != mtx) {
            this->context_ = mtx.get_context();
            this->ld_ = mtx.get_ld();
            this->alloc_elems = mtx.get_alloc_elems();
            this->size = mtx.get_size();
            this->values = mtx.get_values();
            mtx.get_values() = nullptr;
        }
        return *this;
    }

    magma_int_t get_alloc_elems() {
        return alloc_elems;
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

    magma_int_t get_ld() { return ld_; }

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
    magma_int_t ld_;
    magma_int_t alloc_elems = 0;
    dim2 size_;
    value_type* values_ = nullptr;
    ContextType device_;
};


}  // namespace matrix
}  // namespace rls

#endif
