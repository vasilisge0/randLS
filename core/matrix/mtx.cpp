#include "mtx.hpp"
#include "include/base_types.hpp"
#include "../memory/magma_context.hpp"


namespace rls {


template<ContextType device_type>
dim2 Mtx<device_type>::get_size()
{
    return size_;
}

template<ContextType device_type>
size_t Mtx<device_type>::get_num_elems()
{
    return num_elems_;
}

template<ContextType device_type>
std::shared_ptr<Context<device_type>> Mtx<device_type>::get_context() const
{
    return context_;
}

template<ContextType device_type>
void Mtx<device_type>::set_context(std::shared_ptr<Context<device_type>> context)
{
    context_ = context;
}

template<ContextType device_type>
Mtx<device_type>::Mtx(std::shared_ptr<Context<device_type>> context)
{
    context_ = context;
}

template class Mtx<rls::CUDA>;
template class Mtx<rls::CPU>;

}   // end of namespace rls
