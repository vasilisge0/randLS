
#include "../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../core/memory/magma_context.hpp"
#include "base_types.hpp"

namespace rls {
namespace utils {


template <typename value_type_in, typename value_type_out, typename index_type,
          ContextType device_type=CUDA>
void convert(index_type num_rows, index_type num_cols, value_type_in* values_in, index_type ld_in,
             value_type_out* values_out, index_type ld_out);

}
}