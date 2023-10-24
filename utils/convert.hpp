#ifndef RLS_CONVERT_HPP
#define RLS_CONVERT_HPP


#include "../core/memory/magma_context.hpp"
#include "../cuda/preconditioner/preconditioner_kernels.cuh"
#include "base_types.hpp"


namespace rls {
namespace utils {


template <typename value_type_in, typename value_type_out, typename index_type>
void convert(std::shared_ptr<Context<CUDA>> context, index_type num_rows, index_type num_cols, value_type_in* values_in, index_type ld_in,
             value_type_out* values_out, index_type ld_out);

template <typename value_type_in, typename value_type_out, typename index_type>
void convert(std::shared_ptr<Context<CPU>> context, index_type num_rows, index_type num_cols, value_type_in* values_in, index_type ld_in,
             value_type_out* values_out, index_type ld_out);


}   // end of namespace utils
}   // end of namespace rls


#endif
