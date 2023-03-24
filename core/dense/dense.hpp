#ifndef DENSE_HPP
#define DENSE_HPP


#include <iostream>
#include <string>


#include "../../include/base_types.hpp"
#include "../../utils/io.hpp"
#include "../memory/memory.hpp"
#include "../../cuda/solver/lsqr_kernels.cuh"


namespace rls {
namespace matrix {

enum DeviceType{
    CPU,
    GPU
};


template <typename value_type>
class dense {
public:
    dense() {}

    dense(dim2 size_in)
    {
        size = size_in;
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
        std::cout << "size[0] * size[1]: " << size[0] * size[1] << '\n';
        memory::malloc(&values, size[0] * size[1]);
        std::cout << "(values == nullptr): " << (values == nullptr) << '\n';
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
        std::cout << "tmp[0]: " << tmp[0] << '\n';
        std::cout << "size[0]: " << size[0] << ", size[1]: " << size[1] << '\n';
        memory::setmatrix(size[0], size[1], tmp, size[0], values, size[0],
                          queue);
        rls::io::print_mtx_gpu(1, 1, (double*)values, 1, queue);
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

    value_type* get_values() { return values; }

    const value_type* get_const_values() const { return values; }

    dim2& get_size() { return size; }

    ~dense()
    {
        if (values != nullptr) {
            if (device == CPU) {
                memory::free_cpu((magmaDouble_ptr)values);
            }
            else if (device == GPU) {
                memory::free(values);
            }
        }
    }

private:
    DeviceType device;
    dim2 size;
    value_type* values = nullptr;

};


}  // namespace matrix
}  // namespace rls

#endif
