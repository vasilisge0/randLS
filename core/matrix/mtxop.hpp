#ifndef RLS_MTXOP_HPP
#define RLS_MTXOP_HPP


#include "include/base_types.hpp"
#include "../memory/magma_context.hpp"


namespace rls {


template<ContextType device_type>
class MtxOp {
public:

    MtxOp(std::shared_ptr<Context<device_type>> context);

    std::shared_ptr<Context<device_type>> get_context() const;

    void set_context(std::shared_ptr<Context<device_type>> context);

    //template <typename value_type_in>
    //virtual void copy_from(std::shared_ptr<MtxOp<device_type>> mtx) = 0;

    virtual dim2 get_size() = 0;

    virtual size_t get_num_elems() = 0;

private:
    std::shared_ptr<Context<device_type>> context_;
};


}   // end of namespace rls


#endif
