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

enum DeviceType { CPU, GPU };


template <typename value_type>
class dense {
public:
    dense() {}

    dense(std::shared_ptr<Context> context) { context_ = context; }

    dense(std::shared_ptr<Context> context, dim2 size_in) {
        context_ = context;
        size = size_in;
    }

    dense(const dense&& mtx)
    {
        device = mtx.device;
        size = mtx.size;
        values = mtx.values;
        mtx.values = nullptr;
        context_ = mtx.context_;
    }

    void generate(dim2 size_in)
    {
        device = GPU;
        size = size_in;
        memory::malloc(&values, size[0] * size[1]);
    }

    void generate()
    {
        device = GPU;
        memory::malloc(&values, size[0] * size[1]);
    }

    void generate_cpu(dim2 size_in)
    {
        device = CPU;
        size = size_in;
        memory::malloc_cpu(&values, size[0] * size[1]);
    }

    void generate_cpu()
    {
        device = CPU;
        memory::malloc_cpu(&values, size[0] * size[1]);
    }

    void generate(std::string& filename_mtx, magma_queue_t& queue)
    {
        device = GPU;
        io::read_mtx_size((char*)filename_mtx.c_str(), &size[0], &size[1]);
        memory::malloc(&values, size[0] * size[1]);
        value_type* tmp;
        memory::malloc_cpu(&tmp, size[0] * size[1]);
        io::read_mtx_values((char*)filename_mtx.c_str(), size[0], size[1], tmp);
        memory::setmatrix(size[0], size[1], tmp, size[0], values, size[0],
                          queue);
        memory::free_cpu(tmp);
    }

    void generate_cpu(std::string& filename_mtx)
    {
        device = CPU;
        io::read_mtx_size((char*)filename_mtx.c_str(), &size[0], &size[1]);
        memory::malloc_cpu(&values, size[0] * size[1]);
        io::read_mtx_values((char*)filename_mtx.c_str(), size[0], size[1],
                            values);
    }

    void zeros()
    {
        auto num_elems = size[0] * size[1];
        value_type zero = 0.0;
        cuda::set_values(num_elems, zero, values);
    }

    void zeros_cpu()
    {
        if (values != nullptr) {
            auto num_elems = size[0] * size[1];
            for (auto i = 0; i < num_elems; i++) {
                values[i] = 0.0;
            }
        }
    }

    static std::unique_ptr<dense<value_type>> create(std::shared_ptr<Context>
        context) {
        auto tmp = new dense<value_type>(context);
        return std::unique_ptr<dense<value_type>>(tmp);
    }

    static std::unique_ptr<dense<value_type>> create(std::shared_ptr<Context>
        context, dim2 size) {
        auto tmp = new dense<value_type>(context, size);
        return std::unique_ptr<dense<value_type>>(tmp);
    }

    static std::unique_ptr<dense<value_type>> create(
        std::shared_ptr<Context> context, std::string& filename_mtx)
    {
        auto tmp = new dense<value_type>(context);
        auto queue = context->get_queue();
        tmp->generate(filename_mtx, queue);
        return std::unique_ptr<dense<value_type>>(tmp);
    }

    value_type* get_values() { return values; }

    const value_type* get_const_values() const { return values; }

    dim2& get_size() { return size; }

    ~dense()
    {
        if (values != nullptr) {
            if (device == CPU) {
                memory::free_cpu((magmaDouble_ptr)values);
            } else if (device == GPU) {
                memory::free(values);
            }
        }
    }

    std::shared_ptr<Context> get_context() { return context_; }

private:
    DeviceType device;
    dim2 size;
    value_type* values = nullptr;
    std::shared_ptr<Context> context_;
};


}  // namespace matrix
}  // namespace rls

#endif
