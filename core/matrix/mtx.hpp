#ifndef RLS_MTX_HPP
#define RLS_MTX_HPP


#include "mtxop.hpp"
#include "include/base_types.hpp"
#include "../memory/magma_context.hpp"


namespace rls {


template<ContextType device_type>
class Mtx : public MtxOp<device_type>{
public:

    Mtx(std::shared_ptr<Context<device_type>> context);

    std::shared_ptr<Context<device_type>> get_context() const;

    void set_context(std::shared_ptr<Context<device_type>> context);

    dim2 get_size();

    size_t get_num_elems();

protected:
    std::shared_ptr<Context<device_type>> context_;
    dim2 size_;
    size_t num_elems_;
};


}   // end of namespace rls


#endif
