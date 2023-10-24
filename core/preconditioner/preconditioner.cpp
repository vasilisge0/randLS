#include "preconditioner.hpp"


namespace rls {
namespace preconditioner {


Config::Config(double sampling_coefficient)
{
    sampling_coefficient_ = sampling_coefficient;
}

double Config::get_sampling_coefficient() { return sampling_coefficient_; }

void Config::set_sampling_coefficient(double sampling_coefficient)
{
    sampling_coefficient_ = sampling_coefficient;
}

void Config::set_memory_strategy(MemoryStrategy mem_strategy)
{
    mem_strategy_ = mem_strategy;
}


}   // end of namespace preconditioner



preconditioner::logger Preconditioner::get_logger()
{
    return logger_;
}


template <ContextType device_type, typename value_type_apply,
          typename index_type>
PrecondOperator<device_type, value_type_apply, index_type>::PrecondOperator(std::shared_ptr<MtxOp<device_type>> mtx)
{
    mtx_ = mtx;
    context_ = mtx->get_context();
}

template <ContextType device_type, typename value_type_apply,
          typename index_type>
std::shared_ptr<MtxOp<device_type>> PrecondOperator<device_type, value_type_apply, index_type>::get_mtx()
{
    return mtx_;
}

template <ContextType device_type, typename value_type_apply,
          typename index_type>
std::shared_ptr<Context<device_type>> PrecondOperator<device_type, value_type_apply, index_type>::get_context()
{
    return context_;
}


template class PrecondOperator<rls::CUDA, double, magma_int_t>;
template class PrecondOperator<rls::CUDA, float, magma_int_t>;


}   // end of namespace rls
