#include "../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../core/memory/magma_context.hpp"
#include "base_types.hpp"
#include "io.hpp"


namespace rls {
namespace utils {


template <typename value_type_in, typename value_type_out, typename index_type>
void convert(std::shared_ptr<Context<CUDA>> context, index_type num_rows, index_type num_cols, value_type_in* values_in, index_type ld_in,
             value_type_out* values_out, index_type ld_out)
{
    if (typeid(value_type_in) != typeid(value_type_out)) {
        cuda::convert(num_rows, num_cols, values_in, ld_in, values_out, ld_out);
    }
    else
    {
        cuda::convert(num_rows, num_cols, values_in, ld_in, values_out, ld_out);
    }
}

template <typename value_type_in, typename value_type_out, typename index_type>
void convert(std::shared_ptr<Context<CPU>> context, index_type num_rows, index_type num_cols, value_type_in* values_in, index_type ld_in,
             value_type_out* values_out, index_type ld_out)
{

}

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
             float* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
             __half* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
             double* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, float* values_in, magma_int_t ld_in,
             float* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, float* values_in, magma_int_t ld_in,
             double* values_out, magma_int_t ld_out);

//template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
//             double* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, __half* values_in, magma_int_t ld_in,
             double* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, __half* values_in, magma_int_t ld_in,
             float* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, float* values_in, magma_int_t ld_in,
             __half* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, __half* values_in, magma_int_t ld_in,
             __half* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CUDA>> context, int num_rows, int num_cols, int* values_in, int ld_in,
             int* values_out, int ld_out);

//template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
//             double* values_out, magma_int_t ld_out);


template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
             float* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
             __half* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
             double* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, float* values_in, magma_int_t ld_in,
             float* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, float* values_in, magma_int_t ld_in,
             double* values_out, magma_int_t ld_out);

//template void convert(std::shared_ptr<Context<CUDA>> context, magma_int_t num_rows, magma_int_t num_cols, double* values_in, magma_int_t ld_in,
//             double* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, __half* values_in, magma_int_t ld_in,
             double* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, __half* values_in, magma_int_t ld_in,
             float* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, magma_int_t num_rows, magma_int_t num_cols, float* values_in, magma_int_t ld_in,
             __half* values_out, magma_int_t ld_out);

template void convert(std::shared_ptr<Context<CPU>> context, int num_rows, int num_cols, int* values_in, int ld_in,
             int* values_out, int ld_out);

}   // end of namespace utils
}   // end of namespace rls
