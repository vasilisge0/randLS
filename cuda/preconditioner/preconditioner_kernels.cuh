namespace rls {
namespace cuda {


// original
// template <typename value_type_in, typename value_type, typename index_type>
// __host__ void demote(index_type num_rows, index_type num_cols, value_type* mtx,
                    //  index_type ld_mtx, value_type_in* mtx_rp,
                    //  index_type ld_mtx_rp);
// 
// template <typename value_type_in, typename value_type, typename index_type>
// __host__ void promote(index_type num_rows, index_type num_cols,
                    //   value_type_in* mtx, index_type ld_mtx, value_type* mtx_ip,
                    //   index_type ld_mtx_ip);

template <typename value_type_in, typename value_type, typename index_type>
void demote(index_type num_rows, index_type num_cols, value_type* mtx,
                     index_type ld_mtx, value_type_in* mtx_rp,
                     index_type ld_mtx_rp);

template <typename value_type_in, typename value_type, typename index_type>
void promote(index_type num_rows, index_type num_cols,
                      value_type_in* mtx, index_type ld_mtx, value_type* mtx_ip,
                      index_type ld_mtx_ip);

template <typename value_type_in, typename value_type_out, typename index_type>
void convert(index_type num_rows, index_type num_cols,
             value_type_in* mtx_in, index_type ld_in, value_type_out* mtx_out,
             index_type ld_out);

}  // namespace cuda
}  // namespace rls
