#include "../memory/magma_context.hpp"
#include "dense/dense.hpp"
#include "sparse/sparse.hpp"
#include "mtxop.hpp"


namespace rls {


template<ContextType device_type>
MtxOp<device_type>::MtxOp(std::shared_ptr<Context<device_type>> context) { context_ = context; }

template<ContextType device_type>
std::shared_ptr<Context<device_type>> MtxOp<device_type>::get_context() const { return this->context_; }

template<ContextType device_type>
void MtxOp<device_type>::set_context(std::shared_ptr<Context<device_type>> context) {
    context_ = context;
}


template class MtxOp<rls::CUDA>;
template class MtxOp<rls::CPU>;

}   // end of namespace rls
