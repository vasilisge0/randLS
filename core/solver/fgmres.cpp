#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../../cuda/solver/lsqr_kernels.cuh"
#include "../../utils/convert.hpp"
#include "../../utils/io.hpp"
#include "../blas/blas.hpp"
#include "../matrix/dense/dense.hpp"
#include "../matrix/sparse/sparse.hpp"
#include "../memory/memory.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "base_types.hpp"
#include "fgmres.hpp"
#include "solver.hpp"


namespace rls {
namespace solver {
namespace fgmres {


// Specialized gemv operation for Fgmres on the generalized system
// [I A; A' 0] for input m x n matrix A and m >> n.
template <typename device, typename vtype, typename index_type>
void gemv(std::shared_ptr<device> context, magma_trans_t trans,
          index_type num_rows, index_type num_cols, vtype alpha,
          vtype* mtx, index_type ld, vtype* u_vector,
          index_type inc_u, vtype beta, vtype* v_vector,
          index_type inc_v, vtype* tmp)
{
    auto queue = context->get_queue();
    // Dereference the 2 vector parts.
    auto u0 = u_vector;
    auto u1 = &u_vector[num_rows];
    auto v0 = v_vector;
    auto v1 = &v_vector[num_rows];
    // auto tmp_alpha = -alpha;
    auto tmp_alpha = alpha;
    // Compute first part (indices 0:(m-1))
    blas::copy(context, num_rows, v0, inc_v, tmp, inc_v);
    blas::copy(context, num_rows, u0, inc_u, v0, inc_v);
    blas::gemv(context, MagmaNoTrans, num_rows, num_cols, tmp_alpha, mtx,
               num_rows, u1, inc_u, tmp_alpha, v0, inc_v);

    blas::axpy(context, num_rows, beta, tmp, inc_v, v0, inc_v);
    // Compute second part (indices m:m+n)
    blas::gemv(context, MagmaTrans, num_rows, num_cols, (vtype)-1.0, mtx,
               num_rows, u0, inc_u, beta, v1, inc_v);
}

template <typename device, typename vtype>  //, typename index_type>
void gemv(std::shared_ptr<device> context,
          gko::matrix::Dense<vtype>* alpha,
          gko::LinOp* mtx,
          gko::LinOp* mtx_t,
          gko::matrix::Dense<vtype>* u_vector,
          gko::matrix::Dense<vtype>* beta,
          gko::matrix::Dense<vtype>* v_vector)
{
    auto queue = context->get_queue();
    auto size = mtx->get_size();
    // Dereference the 2 vector parts.
    auto exec = context->get_executor();
    auto u0 = gko::matrix::Dense<vtype>::create(
        exec, gko::dim<2>(size[0], 1), gko::make_array_view(exec, size[0], u_vector->get_values()), 1);
    auto u1 = gko::matrix::Dense<vtype>::create(
        exec, gko::dim<2>(size[1], 1), gko::make_array_view(exec, size[1], u_vector->get_values() + size[0]), 1);
    auto v0 = gko::matrix::Dense<vtype>::create(
        exec, gko::dim<2>(size[0], 1), gko::make_array_view(exec, size[0], v_vector->get_values()), 1);
    auto v1 = gko::matrix::Dense<vtype>::create(
        exec, gko::dim<2>(size[1], 1), gko::make_array_view(exec, size[1], v_vector->get_values() + size[0]), 1);
    auto one = gko::initialize<gko::matrix::Dense<vtype>>(
        {(vtype)1.0}, exec);
    v0->scale(beta);
    v0->add_scaled(alpha, u0);
    mtx->apply(alpha, u1, one, v0);
    // Compute second part (indices m:m+n)
    mtx_t->apply(alpha, u0, beta, v1);
}

template void gemv(std::shared_ptr<Context<CUDA>> context,
                   gko::matrix::Dense<double>* alpha, gko::LinOp* mtx,
                   gko::LinOp* mtx_t, gko::matrix::Dense<double>* u_vector,
                   gko::matrix::Dense<double>* beta,
                   gko::matrix::Dense<double>* v_vector);

template void gemv(std::shared_ptr<Context<CUDA>> context,
                   gko::matrix::Dense<float>* alpha, gko::LinOp* mtx,
                   gko::LinOp* mtx_t, gko::matrix::Dense<float>* u_vector,
                   gko::matrix::Dense<float>* beta,
                   gko::matrix::Dense<float>* v_vector);


template <ContextType device, typename vtype, typename index_type>  //, typename index_type>
void gemv(std::shared_ptr<Context<device>> context,
          vtype alpha,
          matrix::Sparse<device, vtype, index_type>* mtx,
          matrix::Sparse<device, vtype, index_type>* mtx_t,
          matrix::Dense<device, vtype>* u_vector,
          vtype beta,
          matrix::Dense<device, vtype>* v_vector)
{
    auto queue = context->get_queue();
    auto size = mtx->get_size();
    // Dereference the 2 vector parts.
    auto exec = context->get_executor();
    auto u0 = matrix::Dense<device, vtype>::create_subcol(u_vector, span(0, size[0]-1), 0);
    auto u1 = matrix::Dense<device, vtype>::create_subcol(u_vector, span(size[0], size[0] + size[1]-1), 0);
    auto v0 = matrix::Dense<device, vtype>::create_subcol(v_vector, span(0, size[0]-1), 0);
    auto v1 = matrix::Dense<device, vtype>::create_subcol(v_vector, span(size[0], size[0] + size[1]-1), 0);
    vtype one = 1.0;
    blas::scale(context, size[0], beta, v0->get_values(), 1);
    blas::axpy(context, size[0], alpha, u0->get_values(), 1, v0->get_values(), 1);
    mtx->apply(alpha, u1.get(), one, v0.get());
    mtx_t->apply(alpha, u0.get(), beta, v1.get());
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
std::shared_ptr<WorkspaceSparse<device, vtype, vtype_internal_0,
                    vtype_precond_0, index_type>>
WorkspaceSparse<device, vtype, vtype_internal_0,
                vtype_precond_0, index_type>::
    create(std::shared_ptr<Context<device>> context,
           std::shared_ptr<MtxOp<device>> mtx,
           std::shared_ptr<
               iterative::FgmresConfig<vtype, vtype_internal_0,
                                       vtype_precond_0, index_type>>
               config,
           dim2 size)
{
    return std::shared_ptr<
        WorkspaceSparse<device, vtype, vtype_internal_0,
                        vtype_precond_0, index_type>>(
        new WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>(context, mtx,
                                                              config, size));
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
WorkspaceSparse<
    device, vtype, vtype_internal_0, vtype_precond_0,
    index_type>::WorkspaceSparse(std::shared_ptr<Context<device>> context,
                                 std::shared_ptr<MtxOp<device>> mtx,
                                 std::shared_ptr<iterative::FgmresConfig<
                                     vtype, vtype_internal_0,
                                     vtype_precond_0, index_type>>
                                     config,
                                 dim2 size)
{
    auto context_cpu = share(Context<CPU>::create());
    context_ = context;
    //max_iter_ = size[1];
    max_iter_ = config->get_iterations();
    auto global_length = size[0] + size[1];
    auto exec_cpu = gko::ReferenceExecutor::create();
    auto exec_gpu = context_->get_executor();
    u = share(dense::create(context_, {global_length, 1}));
    //v_basis = share(dense::create(context_, {(max_iter_ + 1), global_length}));
    //z_basis = share(dense::create(context_, {(max_iter_ + 1), global_length}));
    v_basis = share(dense::create(context_, {global_length, (max_iter_ + 1)}));
    z_basis = share(dense::create(context_, {global_length, (max_iter_ + 1)}));
    w = share(dense::create(context_, {global_length, 1}));
    temp = share(dense::create(context_, {global_length, 1}));
    hessenberg_mtx_gpu =
        share(dense::create(context_, {max_iter_ + 1, max_iter_}));
    hessenberg_rhs_gpu = share(dense::create(context_, {max_iter_ + 1, 1}));
    // CPU
    dim2 st = {max_iter_ + 1, max_iter_};
    hessenberg_mtx = share(dense_cpu::create(context_cpu, st));
    hessenberg_rhs = share(dense_cpu::create(context_cpu, {max_iter_ + 1, 1}));
    std::cout << "\n\n Workspace --> global_length: " << global_length << "\n\n";
    residual = share(dense::create(context_, {global_length, 1}));
    sol0 = share(dense::create(context_, {size[0], 1}));
    if (std::is_same<vtype_internal_0, vtype>::value) {
        w_in = share(dense_internal_0::create(context_, {global_length, 1}));
        v_in =
            share(dense_internal_0::create(context_, {global_length, 1}));
        z_in =
            share(dense_internal_0::create(context_, {global_length, 1}));
        temp_in =
            share(dense_internal_0::create(context_, {global_length, 1}));
        if (auto t = dynamic_cast<matrix::Dense<device, vtype>*>(
                mtx.get());
            t != nullptr) {
            mtx_in = matrix::Dense<device,
                vtype_internal_0>::create(context, {global_length, size[1]});
            static_cast<matrix::Dense<device, vtype_internal_0>*>(mtx_in.get())->copy_from(t);
        } else if (auto t = dynamic_cast<
                       matrix::Sparse<device, vtype, index_type>*>(
                       mtx.get());
                   t != nullptr) {
            //auto t0 = dynamic_cast<
            //    matrix::Sparse<device, vtype, index_type>*>(
            //    mtx_in.get());
            auto t0 = dynamic_cast<
                matrix::Sparse<device, vtype, index_type>*>(
                mtx.get());
            t0->copy_from(t);
            mtx_in_t = rls::share(t->transpose());
        }
    }
    givens_cache =
        std::shared_ptr<std::vector<std::pair<vtype, vtype>>>(
            new std::vector<std::pair<vtype, vtype>>);
    givens_cache->resize(static_cast<size_t>(max_iter_ + 1));
    tmp_cpu = share(dense_cpu::create(context_cpu, {max_iter_ + 1, 1}));
    aug_sol = share(dense::create(context_, {global_length, 1}));
    aug_residual = share(
        dense::create(context_, {global_length, 1}));
    aug_residual->zeros();
    aug_rhs = share(dense::create(context_, {global_length, 1}));
}


template <typename vtype, typename index_type>
void givens_qr(dim2 size, vtype* hessenberg_mtx, index_type ld_h,
               vtype* rhs, index_type cur_iter, index_type max_iter,
               std::vector<std::pair<vtype, vtype>>& givens_cache)
{
    for (index_type row = 0; row < cur_iter + 1; row++) {
        if (row == cur_iter) {
            // Computes givens rotation matrix. The cosine and sine variables,
            // c and s, are then stored in givens_cache for later use.
            const auto alpha = hessenberg_mtx[row + ld_h * cur_iter];
            const auto beta = hessenberg_mtx[(row + 1) + ld_h * cur_iter];
            const auto r = std::hypot(alpha, beta);
            const auto c = alpha / r;
            const auto s = -beta / r;
            givens_cache[row].first = c;
            givens_cache[row].second = s;
            // Updates 2x1 rhs. Entries of rhs are first stored in tmp
            // copies which are then used in the application of the givens
            // rotation matrix.
            const auto tmp_rhs_0 = rhs[row];
            const auto tmp_rhs_1 = rhs[row + 1];
            rhs[row] = c * tmp_rhs_0 - s * tmp_rhs_1;
            rhs[row + 1] = s * tmp_rhs_0 + c * tmp_rhs_1;
        }
        // Updates the current column of the hessenberg matrix. The old
        // values are initially copied into h0 and h1.
        const auto col = cur_iter;
        const auto c = givens_cache[row].first;
        const auto s = givens_cache[row].second;
        const auto h0 = hessenberg_mtx[row + ld_h * col];
        const auto h1 = hessenberg_mtx[(row + 1) + ld_h * col];
        hessenberg_mtx[row + ld_h * col] = c * h0 - s * h1;
        hessenberg_mtx[(row + 1) + ld_h * col] = s * h0 + c * h1;
    }
}

// Input of type Dense, mtx_in, sol_in, rhs_in are the input matrix, solution
// vectors and right-hand side, corresponding to the variables of the input
// problem (not the augmented one). The autmented system, rhs and solution
// vectors are never explicitly formed.
template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
void initialize(
    std::shared_ptr<Context<device>> context, iterative::Logger& logger,
    PrecondOperator<device, vtype, index_type>* precond,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    matrix::Dense<device, vtype>* mtx_in,
    matrix::Dense<device, vtype>* sol_in,
    matrix::Dense<device, vtype>* rhs_in)
{
//    const auto mtx = mtx_in->get_values();
//    const auto rhs = rhs_in->get_values();
//    auto sol = sol_in->get_values();
//    auto residual = workspace->residual->get_values();
//    auto temp = workspace->temp->get_values();
//    auto temp_cpu = workspace->tmp_cpu->get_values();
//    auto v_in = workspace->v_in->get_values();
//    auto hessenberg_rhs = workspace->hessenberg_rhs->get_values();
//    auto size = mtx_in->get_size();
//    auto inc = workspace->inc;
//    const auto global_length = size[0] + size[1];
//    const index_type inc_v = 1;
//    const vtype one = 1.0;
//    const vtype minus_one = -1.0;
//    blas::copy(context, global_length, rhs, inc_v, residual, inc_v);
//    logger.rhsnorm_ = blas::norm2(context, global_length, rhs, 1);
//    fgmres::gemv(context, MagmaNoTrans, size[0], size[1], minus_one, mtx,
//                 size[0], sol, workspace->inc, one, residual, workspace->inc,
//                 temp);
//    precond->apply(context, MagmaTrans, workspace->residual.get());
//    workspace->beta = blas::norm2(context, global_length, residual, inc);
//    temp_cpu[0] = workspace->beta;
//    const auto max_iter = workspace->max_iter_;
//    for (auto i = 0; i < max_iter + 1; i++) {
//        hessenberg_rhs[i] = 0;
//    }
//    hessenberg_rhs[0] = workspace->beta;
//    auto v = workspace->v_basis->get_values();
//    blas::copy(context, global_length, residual, inc, v, inc);
//    blas::scale(context, global_length, (vtype)1.0 / workspace->beta, v,
//                inc);
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
void step_1(
    std::shared_ptr<Context<device>> context,
    std::shared_ptr<iterative::FgmresConfig<vtype, vtype_internal_0,
                                            vtype_precond_0, index_type>> config,
    iterative::Logger& logger,
    PrecondOperator<device, vtype, index_type>* precond,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    matrix::Dense<device, vtype>* mtx_in)
{
//    const auto mtx = mtx_in->get_values();
//    auto size = mtx_in->get_size();
//    auto global_length = size[0] + size[1];
//    auto z = matrix::Dense<device, vtype>::create(context, dim2(global_length, 1),
//        workspace->z_basis->get_values() + global_length * logger.completed_iterations_);
//    auto v = matrix::Dense<device, vtype>::create(context, dim2(global_length, 1),
//        workspace->v_basis->get_values() + global_length * logger.completed_iterations_);
//    auto w = matrix::Dense<device, vtype>::create(context, dim2(global_length, 1),
//        workspace->v_basis->get_values() + global_length * (logger.completed_iterations_ + 1));
//    auto queue = context->get_queue();
//    blas::copy(context, global_length, v->get_values(), 1, z->get_values(), 1);
//    precond->apply(context, MagmaNoTrans, z.get());
//    //if (typeid(vtype_internal_0) != typeid(vtype)) {
//    //    vtype_internal_0 one = 1.0;
//    //    vtype_internal_0 zero = 0.0;
//    //    vtype_internal_0* z_in = workspace->z_in->get_values();
//    //    vtype_internal_0* w_in = workspace->w_in->get_values();
//    //    vtype_internal_0* temp_in = workspace->temp_in->get_values();
//    //    auto t =
//    //        static_cast<matrix::Dense<device, vtype_internal_0>*>(
//    //            workspace->mtx_in.get());
//    //    vtype_internal_0* mtx_in = t->get_values();
//    //    rls::utils::convert(context, global_length, 1, z, global_length, z_in,
//    //                        global_length);
//    //    rls::utils::convert(context, global_length, 1, w, global_length, w_in,
//    //                        global_length);
//    //    // fix precisions issues
//    //    fgmres::gemv(context, MagmaNoTrans, size[0], size[1], one, mtx_in,
//    //                 size[0], z_in, 1, zero, w_in, 1, temp_in);
//    //    rls::utils::convert(context, global_length, 1, w_in, global_length, w,
//    //                        global_length);
//    //} else {
//        vtype one = 1.0;
//        vtype zero = 0.0;
//        fgmres::gemv(context, MagmaTrans, size[0], size[1], one, mtx,
//                     size[0], z->get_values(), 1, zero, w->get_values(), 1,
//                     workspace->temp->get_values());
//    //}
//    precond->apply(context, MagmaTrans, w.get());
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
void step_2(
    std::shared_ptr<Context<device>> context,
    std::shared_ptr<iterative::FgmresConfig<vtype, vtype_internal_0,
                                            vtype_precond_0, index_type>>
        config,
    iterative::Logger logger,
    PrecondOperator<device, vtype, index_type>* precond,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    matrix::Dense<device, vtype>* mtx_in,
    matrix::Dense<device, vtype>* sol_in)
{
//    const auto mtx = mtx_in->get_values();
//    auto sol = sol_in->get_values();
//    auto size = mtx_in->get_size();
//    auto global_length = size[0] + size[1];
//    const auto max_iter = config->get_iterations();
//    auto ld = max_iter + 1;
//    auto hessenberg_mtx = workspace->hessenberg_mtx->get_values();
//    auto hessenberg_rhs = workspace->hessenberg_rhs->get_values();
//    const dim2 size_h = {max_iter + 1, max_iter};
//    auto w = workspace->v_basis->get_values() +
//             global_length * (logger.completed_iterations_ + 1);
//    const auto cur_iter = logger.completed_iterations_;
//    auto queue = context->get_queue();
//    if (logger.completed_iterations_ == 0) {
//        for (auto j = 0; j < max_iter; j++) {
//            for (auto i = 0; i < max_iter + 1; i++) {
//                hessenberg_mtx[i + ld * j] = 0.0;
//            }
//        }
//    }
//    for (index_type i = 0; i < logger.completed_iterations_ + 1; i++) {
//        auto v = workspace->v_basis->get_values() + global_length * i;
//        hessenberg_mtx[i + ld * cur_iter] =
//            blas::dot(context, global_length, v, 1, w, 1);
//        blas::axpy(context, global_length, -hessenberg_mtx[i + ld * cur_iter],
//                   v, 1, w, 1);
//    }
//    const auto w_norm = blas::norm2(context, global_length, w, 1);
//    magma_queue_sync(queue);
//    hessenberg_mtx[cur_iter + 1 + ld * cur_iter] = w_norm;
//    workspace->h = w_norm;
//    // Solve here using givens qr.
//    fgmres::givens_qr(size, hessenberg_mtx, (max_iter + 1), hessenberg_rhs,
//                      cur_iter, max_iter, *workspace->givens_cache.get());
//    magma_queue_sync(queue);
//    const auto hessenberg_mtx_gpu = workspace->hessenberg_mtx_gpu;
//    const auto hessenberg_rhs_gpu = workspace->hessenberg_rhs_gpu;
//    memory::setmatrix(max_iter + 1, max_iter, hessenberg_mtx, max_iter + 1,
//                      hessenberg_mtx_gpu->get_values(), max_iter + 1, queue);
//    memory::setmatrix(max_iter + 1, 1, hessenberg_rhs, max_iter + 1,
//                      hessenberg_rhs_gpu->get_values(), max_iter + 1, queue);
//    blas::trsv(context, MagmaUpper, MagmaNoTrans, MagmaNonUnit, cur_iter + 1,
//               hessenberg_mtx_gpu->get_values(), (max_iter + 1),
//               hessenberg_rhs_gpu->get_values(), 1);
//    magma_queue_sync(queue);
//    // Update solution x.
//    // Needs to be changed as to augment an initial solution vector.
//    blas::gemv(context, MagmaNoTrans, global_length, cur_iter + 1,
//               (vtype)1.0, workspace->z_basis->get_values(), global_length,
//               hessenberg_rhs_gpu->get_values(), 1, (vtype)0.0, sol, 1);
}

template <typename vtype, typename vtype_internal_0,
          typename vtype_precond_0, typename index_type,
          ContextType device>
bool check_stopping_criteria(
    std::shared_ptr<Context<device>> context,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    std::shared_ptr<iterative::FgmresConfig<vtype, vtype_internal_0,
                                            vtype_precond_0, index_type>>
        config,
    iterative::Logger& logger, matrix::Dense<device, vtype>* mtx_in,
    matrix::Dense<device, vtype>* rhs_in,
    matrix::Dense<device, vtype>* sol_in)
{
//    auto num_rows = mtx_in->get_size()[0];
//    auto num_cols = mtx_in->get_size()[1];
//    auto mtx = mtx_in->get_values();
//    auto rhs = rhs_in->get_values();
//    auto sol = sol_in->get_values();
//    auto global_length = num_rows + num_cols;
//    auto res_vector = workspace->residual->get_values();
//    auto temp = workspace->temp->get_values();
//    auto inc = workspace->inc;
//    logger.completed_iterations_ += 1;
//    memory::zeros<vtype, CUDA>({num_cols, 1}, &res_vector[num_rows]);
//    blas::copy(context, num_rows, rhs, 1, res_vector, 1);
//    const vtype one = 1.0;
//    const vtype minus_one = -1.0;
//    fgmres::gemv(context, MagmaNoTrans, num_rows, num_cols, minus_one, mtx,
//                 num_rows, sol, inc, one, res_vector, inc, temp);
//    logger.resnorm_ = blas::norm2(context, global_length, res_vector, 1);
//    logger.resnorm_ = logger.resnorm_ / logger.rhsnorm_;
//    if ((logger.completed_iterations_ >= config->get_iterations()) ||
//        (logger.resnorm_ < config->get_tolerance())) {
//        std::cout << ">>> logger.resnorm_: " << logger.resnorm_ << ", iter: " << logger.completed_iterations_ << "\n";
//        return true;
//    } else {
//        return false;
//    }
//    return true;
}

template struct WorkspaceSparse<CUDA, double, double, double, magma_int_t>;
template struct WorkspaceSparse<CUDA, double, float, double, magma_int_t>;
// template struct WorkspaceSparse<CUDA, double, __half, double, magma_int_t>;
template struct WorkspaceSparse<CUDA, double, float, float, magma_int_t>;
// template struct WorkspaceSparse<CUDA, double, __half, __half, magma_int_t>;
template struct WorkspaceSparse<CUDA, float, float, float, magma_int_t>;
// template struct WorkspaceSparse<CUDA, float, __half, float, magma_int_t>;

template struct WorkspaceSparse<CPU, double, double, double, magma_int_t>;


template <ContextType device,         // Device type of context.
          typename vtype,             // Value type of input precision.
          typename vtype_internal_0,  // Internal precision.
          typename vtype_precond_0,   // Apply preconditioner precision.
          typename index_type>
void initialize(
    std::shared_ptr<Context<device>> context,  // Library context
    iterative::Logger*
        logger,  // Logger for keeping track of iterations and residual norm.
    PrecondOperator<device, vtype, index_type>* precond,
    fgmres::WorkspaceSparse<
        device, vtype,
        vtype_internal_0,  // Preconditioner operation (uses
                                // preconditioner-apply precision)
        vtype_precond_0, index_type>*
        workspace,  // Workspace holding all memory used by fgmres.
    matrix::Sparse<device, vtype, index_type>*
        mtx_in,  // Input matrix (sparse).
    matrix::Sparse<device, vtype, index_type>*
        mtx_t,  // Input matrix (sparse).
    //gko::LinOp* mtx_t,  // Input matrix (sparse).
    matrix::Dense<device, vtype>*
        sol_in,  // Solution vector (dense) <-- change those.
    matrix::Dense<device, vtype>*
        rhs_in)  // Rhs vector (dense) <-- change those.
{
    workspace->completed_iterations_per_restart = 0;
    vtype one = 1.0;
    vtype minus_one = -1.0;
    auto r_mtx = workspace->residual.get();
    workspace->residual->copy_from(rhs_in);
    workspace->rhsnorm = blas::norm2(context, r_mtx->get_size()[0], r_mtx->get_values(), 1);
    fgmres::gemv(context, minus_one, mtx_in,
                 mtx_t, sol_in, one, r_mtx);
    precond->apply(context, MagmaTrans, r_mtx);
    workspace->beta = blas::norm2(context, workspace->residual->get_size()[0], workspace->residual->get_values(), 1);
    workspace->tmp_cpu->get_values()[0] = workspace->beta;
    const auto max_iter = workspace->max_iter_;
    for (auto i = 0; i < max_iter + 1; i++) {
        workspace->hessenberg_rhs->get_values()[i] = 0;
    }
    workspace->hessenberg_rhs->get_values()[0] = workspace->beta;
    auto v = matrix::Dense<device, vtype>::create_submatrix(workspace->v_basis.get(), span(0, 0));
    v->copy_from(r_mtx);
    blas::scale(context, v->get_size()[0], 1 / workspace->beta, v->get_values(), 1);
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
void step_1(
    std::shared_ptr<Context<device>> context,
    iterative::FgmresConfig<vtype, vtype_internal_0,
                                            vtype_precond_0, index_type>*
        config,
    iterative::Logger* logger,
    PrecondOperator<device, vtype, index_type>* precond,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    matrix::Sparse<device, vtype, index_type>* mtx,
    matrix::Sparse<device, vtype, index_type>* mtx_t)
{
    auto cur_iter = workspace->completed_iterations_per_restart;
    auto z = matrix::Dense<device, vtype>::create_submatrix(workspace->z_basis.get(), span(cur_iter, cur_iter));
    auto v = matrix::Dense<device, vtype>::create_submatrix(workspace->v_basis.get(), span(cur_iter, cur_iter));
    auto w = matrix::Dense<device, vtype>::create_submatrix(workspace->v_basis.get(), span(cur_iter + 1, cur_iter + 1));
    auto queue = context->get_queue();
    vtype one = 1.0;
    vtype zero = 0.0;
    z->copy_from(v.get());
    precond->apply(context, MagmaNoTrans, z.get());
    fgmres::gemv(context, one, mtx, mtx_t, z.get(), zero, w.get());
    precond->apply(context, MagmaTrans, w.get());
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
void step_2(
    std::shared_ptr<Context<device>> context,
    iterative::FgmresConfig<
        vtype, vtype_internal_0,
        vtype_precond_0, index_type>*
        config,
    iterative::Logger* logger,
    PrecondOperator<device, vtype, index_type>*
        precond,
    fgmres::WorkspaceSparse<device, vtype,
                            vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    matrix::Sparse<device, vtype, index_type>* mtx,
    matrix::Sparse<device, vtype, index_type>* mtx_t,
    matrix::Dense<device, vtype>* sol)
{
    auto queue = context->get_queue();
    auto exec = context->get_executor();
    auto size = mtx->get_size();
    auto global_length = size[0] + size[1];
    const auto max_iter =
        config->get_iterations();
    auto ld = max_iter + 1;
    auto hessenberg_mtx = workspace->hessenberg_mtx->get_values();
    const dim2 size_h = {max_iter + 1, max_iter};
    const auto cur_iter = workspace->completed_iterations_per_restart;
    auto w = matrix::Dense<device, vtype>::create_submatrix(workspace->v_basis.get(),
        span(cur_iter + 1, cur_iter + 1));

    // Orthogonalize.
    if (cur_iter == 0) {
        for (auto j = 0; j < max_iter; j++) {
            for (auto i = 0; i < max_iter + 1; i++) {
                hessenberg_mtx[i + ld * j] = 0.0;
            }
        }
    }
    vtype zero = 0.0;
    vtype minus_one = -1.0;
    for (index_type i = 0; i < cur_iter + 1; i++) {
        auto v = matrix::Dense<device, vtype>::create_submatrix(workspace->v_basis.get(), span(i, i));
        auto t = blas::dot(context, global_length, v->get_values(), 1, w->get_values(), 1);
        hessenberg_mtx[i + ld * cur_iter] = t;
        blas::axpy(context, v->get_size()[0], -t, v->get_values(), 1, w->get_values(), 1);
    }
    const auto w_norm = blas::norm2(context, w->get_size()[0], w->get_values(), 1);
    workspace->h = w_norm;
    hessenberg_mtx[cur_iter + 1 + (max_iter + 1) * cur_iter] = w_norm;

    // Solve here using givens qr.
    fgmres::givens_qr(dim2(size[0], size[1]), hessenberg_mtx, (max_iter + 1),
                      workspace->hessenberg_rhs->get_values(), cur_iter,
                      max_iter, *workspace->givens_cache.get());
    const auto hessenberg_mtx_gpu = workspace->hessenberg_mtx_gpu->get_values();
    const auto hessenberg_rhs_gpu = workspace->hessenberg_rhs_gpu->get_values();
    memory::setmatrix(max_iter + 1, max_iter, hessenberg_mtx,
                      max_iter + 1, hessenberg_mtx_gpu, max_iter + 1,
                      queue);
    memory::setmatrix(max_iter + 1, 1, workspace->hessenberg_rhs->get_values(),
                      max_iter + 1, hessenberg_rhs_gpu, max_iter + 1,
                      queue);

    blas::trsv(context, MagmaUpper, MagmaNoTrans, MagmaNonUnit, cur_iter + 1,   // Solves with factorized matrix.
        hessenberg_mtx_gpu, (max_iter + 1), hessenberg_rhs_gpu, 1);

    // Update solution x. Needs to be changed as to augment an initial
    // solution vector.
    blas::gemv(context, MagmaNoTrans, (int)global_length, (int)cur_iter + 1,
        (vtype)1.0, workspace->z_basis->get_values(),
        (int)global_length, hessenberg_rhs_gpu, (int)1, (vtype)0.0,
        sol->get_values(), (int)1);

    // works up to here
}

template <typename vtype, typename vtype_internal_0,
          typename vtype_precond_0, typename index_type,
          ContextType device>
bool check_stopping_criteria(
    std::shared_ptr<Context<device>> context,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    iterative::FgmresConfig<vtype, vtype_internal_0,
                                            vtype_precond_0, index_type>*
        config,
    iterative::Logger* logger,
    matrix::Sparse<device, vtype, index_type>* mtx_in,
    matrix::Sparse<device, vtype, index_type>* mtx_t,
    matrix::Dense<device, vtype>* rhs_in,
    matrix::Dense<device, vtype>* sol_in)
{
    workspace->completed_iterations += 1;
    workspace->completed_iterations_per_restart += 1;
    vtype one = 1.0;
    vtype minus_one = -1.0;
    //workspace->residual->copy_from(rhs_in);
    auto v = matrix::Dense<device, vtype>::create_submatrix(workspace->residual.get(), span(0, 0));
    v->copy_from(rhs_in);
    //fgmres::gemv(context, minus_one, mtx_in, mtx_t,
    //             sol_in, one, workspace->residual.get());
    fgmres::gemv(context, minus_one, mtx_in, mtx_t,
                 sol_in, one, v.get());
    auto queue = context->get_queue();
    auto t = blas::norm2(context, workspace->residual->get_size()[0], workspace->residual->get_values(), 1);
    workspace->resnorm = t / workspace->rhsnorm;    // save workspace->rhsnorm
    std::cout << "completed_iterations: " << workspace->completed_iterations << " / " << workspace->resnorm << '\n';
    if ((workspace->completed_iterations >= config->get_iterations()) ||
         (workspace->resnorm < config->get_tolerance())) {
         return true;
    } else {
        return false;
    }
    return true;
}

}  // end namespace fgmres


template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
std::unique_ptr<Fgmres<device, vtype, vtype_internal_0,
                              vtype_precond_0, index_type>>
Fgmres<device, vtype, vtype_internal_0, vtype_precond_0,
       index_type>::
    create(std::shared_ptr<Context<device>> context,
           std::shared_ptr<iterative::Config> config,
           std::shared_ptr<MtxOp<device>> mtx,
           std::shared_ptr<matrix::Dense<device, vtype>> sol,
           std::shared_ptr<matrix::Dense<device, vtype>> rhs)
{
    auto t = std::static_pointer_cast<iterative::FgmresConfig<
        vtype, vtype_internal_0, vtype_precond_0, index_type>>(
        config);
    return std::unique_ptr<
        Fgmres<device, vtype, vtype_internal_0,
               vtype_precond_0, index_type>>(
        new Fgmres<device, vtype, vtype_internal_0,
                   vtype_precond_0, index_type>(context, t, mtx, sol,
                                                     rhs));
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
Fgmres<device, vtype, vtype_internal_0, vtype_precond_0,
       index_type>::
    Fgmres(std::shared_ptr<Context<device>> context,
           std::shared_ptr<
               iterative::FgmresConfig<vtype, vtype_internal_0,
                                       vtype_precond_0, index_type>>
               config,
           std::shared_ptr<MtxOp<device>> mtx,
           std::shared_ptr<matrix::Dense<device, vtype>> sol,
           std::shared_ptr<matrix::Dense<device, vtype>> rhs)
    : Solver<device>(mtx->get_context())
{
    this->mtx_ = mtx;
    this->rhs_ = rhs;
    this->sol_ = sol;
    workspace_ =
        fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                                vtype_precond_0,
                                index_type>::create(context, mtx, config,
                                                    mtx->get_size());
    workspace_->rhsnorm = blas::norm2(context, rhs_->get_size()[0], rhs_->get_values(), 1);
    //workspace_->rhsnorm = blas::norm2(context, workspace_->aug_rhs->get_size()[0], workspace_->aug_rhs->get_values(), 1);
    blas::copy(context, mtx->get_size()[1], this->sol_->get_values(), 1,
               workspace_->aug_sol->get_values() + mtx->get_size()[0], 1);
    config_ = config;
    blas::copy(context, rhs->get_size()[0], rhs->get_values(), 1,
               workspace_->aug_rhs->get_values(), 1);
    logger_ = iterative::Logger::create(config_.get());
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
bool Fgmres<device, vtype, vtype_internal_0, vtype_precond_0,
       index_type>::stagnates()
{
    //return logger_.solver_stagnates(config_->get_stagnation_tolerance(),
    //    config_->get_stagnation_weight());
    return false;
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
bool Fgmres<device, vtype, vtype_internal_0, vtype_precond_0,
       index_type>::converges()
{
    //return logger_.solver_stagnates(config_->get_stagnation_tolerance(),
    //    config_->get_stagnation_weight());
    //return false;
//    auto logger = config_->get_logger();
//    return (logger.resnorm_ < config_->get_tolerance());
    return (workspace_->resnorm < config_->get_tolerance());
}


template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
extern void run_fgmres(
    std::shared_ptr<Context<device>> context,
    std::shared_ptr<iterative::FgmresConfig<vtype, vtype_internal_0,
                                            vtype_precond_0, index_type>>
        config,
    iterative::Logger& logger,
    PrecondOperator<device, vtype, index_type>* precond,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>* workspace,
    matrix::Dense<device, vtype>* mtx,
    matrix::Dense<device, vtype>* rhs,
    matrix::Dense<device, vtype>* sol)
{
//    auto solin = sol->get_values();
//    auto size = mtx->get_size();
//    auto global_length = size[0] + size[1];
//    fgmres::initialize(context, logger, precond, workspace, mtx, sol, rhs);
//    while (1) {
//        fgmres::step_1(context, config, logger, precond, workspace, mtx);
//        fgmres::step_2(context, config, logger, precond, workspace, mtx, sol);
//        if (fgmres::check_stopping_criteria(context, workspace, config, logger,
//                                            mtx, rhs, sol)) {
//            break;
//        }
//        blas::scale(context, global_length, (vtype)1.0 / workspace->h,
//                    workspace->v_basis->get_values() +
//                        global_length * (logger.completed_iterations_),
//                    workspace->inc);
//    }
}

template void run_fgmres(
    std::shared_ptr<Context<CUDA>> context,
    std::shared_ptr<
        iterative::FgmresConfig<double, double, double, magma_int_t>>
        config,
    iterative::Logger& logger,
    PrecondOperator<CUDA, double, magma_int_t>* precond,
    fgmres::WorkspaceSparse<CUDA, double, double, double, magma_int_t>*
        workspace,
    matrix::Dense<CUDA, double>* mtx, matrix::Dense<CUDA, double>* rhs,
    matrix::Dense<CUDA, double>* sol);

template void run_fgmres(
    std::shared_ptr<Context<CUDA>> context,
    std::shared_ptr<iterative::FgmresConfig<double, float, float, magma_int_t>>
        config,
    iterative::Logger& logger,
    PrecondOperator<CUDA, double, magma_int_t>* precond,
    fgmres::WorkspaceSparse<CUDA, double, float, float, magma_int_t>* workspace,
    matrix::Dense<CUDA, double>* mtx, matrix::Dense<CUDA, double>* rhs,
    matrix::Dense<CUDA, double>* sol);

// template void run_fgmres(
//     std::shared_ptr<Context<CUDA>> context,
//     std::shared_ptr<iterative::FgmresConfig<double, __half,
//                             __half, magma_int_t>> config,
//     iterative::Logger& logger,
//     PrecondOperator<CUDA, __half, magma_int_t>* precond,
//     fgmres::WorkspaceSparse<CUDA, double, __half, __half, magma_int_t>*
//     workspace, matrix::Dense<CUDA, double>* mtx, matrix::Dense<CUDA, double>*
//     rhs, matrix::Dense<CUDA, double>* sol);

template void run_fgmres(
    std::shared_ptr<Context<CUDA>> context,
    std::shared_ptr<iterative::FgmresConfig<float, float, float, magma_int_t>>
        config,
    iterative::Logger& logger,
    PrecondOperator<CUDA, float, magma_int_t>* precond,
    fgmres::WorkspaceSparse<CUDA, float, float, float, magma_int_t>* workspace,
    matrix::Dense<CUDA, float>* mtx, matrix::Dense<CUDA, float>* rhs,
    matrix::Dense<CUDA, float>* sol);

// template void run_fgmres(
//     std::shared_ptr<Context<CUDA>> context,
//     std::shared_ptr<iterative::FgmresConfig<float, __half,
//                             __half, magma_int_t>> config,
//     iterative::Logger& logger,
//     PrecondOperator<CUDA, __half, magma_int_t>* precond,
//     fgmres::WorkspaceSparse<CUDA, float, __half, __half, magma_int_t>*
//     workspace, matrix::Dense<CUDA, float>* mtx, matrix::Dense<CUDA, float>*
//     rhs, matrix::Dense<CUDA, float>* sol);

/* Definitions of Fgmres class methods */

template class Fgmres<CUDA, double, double, double, magma_int_t>;
template class Fgmres<CUDA, double, float, double, magma_int_t>;
template class Fgmres<CUDA, double, double, float, magma_int_t>;
// template class Fgmres<CUDA, double, __half, double, magma_int_t>;
template class Fgmres<CUDA, double, float, float, magma_int_t>;
// template class Fgmres<CUDA, double, __half, __half, magma_int_t>;
template class Fgmres<CUDA, float, float, float, magma_int_t>;
// template class Fgmres<CUDA, float, __half, float, magma_int_t>;


template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
//extern
void run_fgmres(
    std::shared_ptr<Context<device>>
        context,
    iterative::FgmresConfig<
        vtype, vtype_internal_0,
        vtype_precond_0, index_type>*
        config,
    iterative::Logger* logger,
    PrecondOperator<device, vtype, index_type>*
        precond,
    fgmres::WorkspaceSparse<device, vtype, vtype_internal_0,
                            vtype_precond_0, index_type>*
        workspace,
    matrix::Sparse<device, vtype, index_type>*
        mtx,                                      // Sparse matrix (original).
    matrix::Dense<device, vtype>* rhs,  // Rhs vector (augmented).
    matrix::Dense<device, vtype>* sol)  // Solution vector (augmented).
{
    auto exec = context->get_executor();     // Ginkgo Executor.
    auto global_length = sol->get_size()[0]; // Length of the augmented system.
    auto mtx_t =
        dynamic_cast<matrix::Sparse<device, vtype, index_type>*>(
            workspace->mtx_in_t.get());
    fgmres::initialize(
        context,
        logger, precond, workspace, mtx, mtx_t, sol, rhs);
    while (1) {
        fgmres::step_1(context,
                       config, logger, precond, workspace, mtx, mtx_t);
        fgmres::step_2(context, config, logger, precond, workspace, mtx, mtx_t, sol);
        if (fgmres::check_stopping_criteria(context, workspace, config, logger, mtx,
                                            mtx_t, rhs, sol)) {
            // have to do some copying here
            break;
        }
        auto cur_iter = workspace->completed_iterations_per_restart;
        // Vector w to be scaled.
        auto w = matrix::Dense<device, vtype>::create_submatrix(workspace->v_basis.get(),
            span(cur_iter, cur_iter));
        blas::scale(context, w->get_size()[0], 1/workspace->h, w->get_values(), 1);
    }
}

template void run_fgmres(
    std::shared_ptr<Context<CUDA>>
        context,
    iterative::FgmresConfig<
        double, double, double, magma_int_t>*
        config,
    iterative::Logger* logger,
    PrecondOperator<CUDA, double, magma_int_t>*
        precond,
    fgmres::WorkspaceSparse<CUDA, double, double, double, magma_int_t>*
        workspace,
    matrix::Sparse<CUDA, double, magma_int_t>* mtx, // Sparse matrix (original).
    matrix::Dense<CUDA, double>* rhs,   // Rhs vector (augmented).
    matrix::Dense<CUDA, double>* sol);  // Solution vector (augmented).


template void run_fgmres(
    std::shared_ptr<Context<CUDA>>
        context,
    iterative::FgmresConfig<
        double, double, float, magma_int_t>*
        config,
    iterative::Logger* logger,
    PrecondOperator<CUDA, double, magma_int_t>*
        precond,
    fgmres::WorkspaceSparse<CUDA, double, double, float, magma_int_t>*
        workspace,
    matrix::Sparse<CUDA, double, magma_int_t>* mtx, // Sparse matrix (original).
    matrix::Dense<CUDA, double>* rhs,   // Rhs vector (augmented).
    matrix::Dense<CUDA, double>* sol);  // Solution vector (augmented).

template void run_fgmres(
    std::shared_ptr<Context<CUDA>>
        context,
    iterative::FgmresConfig<
        float, float, float, magma_int_t>*
        config,
    iterative::Logger* logger,
    PrecondOperator<CUDA, float, magma_int_t>*
        precond,
    fgmres::WorkspaceSparse<CUDA, float, float, float, magma_int_t>*
        workspace,
    matrix::Sparse<CUDA, float, magma_int_t>* mtx, // Sparse matrix (original).
    matrix::Dense<CUDA, float>* rhs,   // Rhs vector (augmented).
    matrix::Dense<CUDA, float>* sol);  // Solution vector (augmented).

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
void Fgmres<device, vtype, vtype_internal_0,
            vtype_precond_0, index_type>::run_with_logger()
{
    auto logger = config_->get_logger();
    //if (config_->use_precond_) {
    //    for (auto i = 0; i < this->logger_.warmup_runs_; i++) {
    //        sol_->zeros();
    //        // run_fgmres(this->context_, this->logger_,
    //        //            static_cast<SketchOperator<
    //        //                device, vtype_precond_0, index_type>*>(
    //        //                config_.precond_),
    //        //            workspace_, mtx_, rhs_, sol_);
    //    }
    //    this->logger_.runtime_ = 0.0;
    //    this->logger_.completed_iterations_ = 0;
    //    this->logger_.resnorm_ = 0.0;
    //    for (auto i = 0; i < this->logger_.runs_; i++) {
    //        sol_->zeros();
    //        run_fgmres(
    //            this->context_, this->logger_,
    //            static_cast<SketchOperator<device, vtype_precond_0,
    //                                       index_type>*>(config_.precond_),
    //            workspace_, mtx_, rhs_, sol_);
    //    }
    //    this->logger_.runtime_ = this->logger_.runtime_ / this->logger_.runs_;
    //    this->iter_ = this->iter_ / this->logger_.runs_;
    //    this->resnorm_ = this->resnorm_ / this->logger_.runs_;
    //} else {
    //    // Run non-preconditioned FGMRES.
    //}
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
void Fgmres<device, vtype, vtype_internal_0,
            vtype_precond_0, index_type>::run()
{
    auto context = this->get_context();
    auto precond = config_->get_precond();
    if (auto t =
            dynamic_cast<matrix::Dense<device, vtype>*>(mtx_.get());
        t != nullptr) {
        // run_fgmres(context, this->config_, logger,
        //     static_cast<PrecondOperator<device, vtype,
        //                               index_type>*>(precond.get()),
        //     workspace_.get(), t, workspace_->aug_rhs.get(),
        //     workspace_->aug_sol.get());
    } else if (auto t = dynamic_cast<matrix::Sparse<device, vtype, index_type>*>(
                   mtx_.get());
               t != nullptr) {
        run_fgmres(
            context, this->config_.get(), logger_.get(),
            static_cast<PrecondOperator<device, vtype,
                                        index_type>*>(precond.get()),
            workspace_.get(), t, workspace_->aug_rhs.get(),
            workspace_->aug_sol.get());
    } else {
        // Run non-preconditioned Fgmres.
    }
}

template <ContextType device, typename vtype,
          typename vtype_internal_0, typename vtype_precond_0,
          typename index_type>
iterative::Logger Fgmres<device, vtype, vtype_internal_0, vtype_precond_0, index_type>::get_logger()
{
    //return this->config_->get_logger();
}

}  // end of namespace solver
}  // end of namespace rls
