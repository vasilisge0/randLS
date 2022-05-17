#ifndef DENSE_HPP
#define DENSE_HPP


#include <iostream>
#include <string>


#include "../../include/base_types.hpp"
#include "../../utils/io.hpp"
#include "../memory/memory.hpp"


namespace rls {
namespace matrix {


template <typename value_type>
class dense {
public:
    dense() {}

    dense(dim2 size_in)
    {
        size = size_in;
        memory::malloc(&values, size[0] * size[1]);
    }

    ~dense()
    {
        if (values != nullptr) {
            memory::free(values);
        }
    }

    void generate(dim2 size_in)
    {
        size = size_in;
        memory::malloc(&values, size[0] * size[1]);
    }

    void generate_cpu(dim2 size_in)
    {
        size = size_in;
        memory::malloc_cpu(&values, size[0] * size[1]);
    }

    void generate(std::string& filename_mtx, magma_queue_t& queue)
    {
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
        io::read_mtx_size((char*)filename_mtx.c_str(), &size[0], &size[1]);
        memory::malloc_cpu(&values, size[0] * size[1]);
        io::read_mtx_values((char*)filename_mtx.c_str(), size[0], size[1],
                            values);
    }

    void zeros()
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

private:
    dim2 size;
    value_type* values = nullptr;
};


}  // namespace matrix
}  // namespace rls

#endif
