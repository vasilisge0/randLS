#include "../memory/magma_context.hpp"
#include "dense/dense.hpp"
#include "sparse/sparse.hpp"
#include "mtxop.hpp"


namespace rls {


template class MtxOp<rls::CUDA>;
template class MtxOp<rls::CPU>;

}   // end of namespace rls
