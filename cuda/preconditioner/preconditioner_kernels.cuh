namespace rls {
namespace cuda {


template <typename value_type_in, typename value_type, typename index_type>
__host__ void demote(index_type num_rows, index_type num_cols, value_type* mtx,
                     index_type ld_mtx, value_type_in* mtx_rp,
                     index_type ld_mtx_rp);

template <typename value_type_in, typename value_type, typename index_type>
__host__ void promote(index_type num_rows, index_type num_cols,
                      value_type_in* mtx, index_type ld_mtx, value_type* mtx_ip,
                      index_type ld_mtx_ip);


}  // namespace cuda
}  // namespace rls
