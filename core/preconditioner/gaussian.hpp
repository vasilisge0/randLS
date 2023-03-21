#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP
#include "../../utils/io.hpp"


#include "../memory/detail.hpp"
#include "../dense/dense.hpp"
#include "preconditioner.hpp"
#include <memory>
#include "magma_v2.h"
#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../blas/blas.hpp"

#include <iostream>


namespace rls {
namespace preconditioner {

;
// template <typename value_type_in, typename value_type,
        //   typename index_type>
// void compute_precond(index_type num_rows_sketch, index_type num_cols_sketch, value_type* dsketch, index_type ld_sketch,
            //   index_type num_rows_mtx, index_type num_cols_mtx, value_type* dmtx, index_type ld_mtx,
            //   value_type* dr_factor, index_type ld_r_factor,
            //   value_type* hat_mtx, detail::magma_info& info);

template <typename value_type_in, typename value_type,
          typename index_type>
void compute_precond(index_type num_rows_sketch, index_type num_cols_sketch,
              value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx,
              value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
              index_type ld_r_factor, value_type* hat_mtx,
              detail::magma_info& info, double* runtime);

template<typename value_type_in, typename value_type, typename index_type>
class gaussian : public preconditioner<value_type_in, value_type, index_type> {
public:
    // void apply(matrix::dense<value_type>& mtx, matrix::dense<value_type>& u);

    // void apply(matrix::dense<value_type>& mtx, matrix::dense<value_type_in>& mtx_in,
    //            matrix::dense<value_type>& u, matrix::dense<value_type_in>& u_in);

    gaussian() { 
        this->set_type();
        this->precond_type = Gaussian;
        this->sampling_coeff = 2.0;
    }

    gaussian(detail::magma_info& magma_config) { 
        this->set_type();
        this->precond_type = Gaussian;
        this->sampling_coeff = 2.0;
        this->info = magma_config;
    }

    gaussian(matrix::dense<value_type>& sketch_mtx_in) {
        sketch_mtx = std::move(sketch_mtx);
        this->set_type();
        // allocate(size, sketch_mtx.get_size()[0]);
    }

    gaussian(std::shared_ptr<matrix::dense<value_type>> mtx_in,
        std::shared_ptr<matrix::dense<value_type>> sketch_mtx_in) {

        this->set_type();
        this->mtx = mtx_in;
        sketch_mtx = sketch_mtx_in;
        auto size = this->mtx->get_size();

        auto sampled_rows = sketch_mtx->get_size()[0];
        precond_mtx = std::shared_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>());
        precond_mtx_internal = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        dt = std::unique_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>());
        mtx_rp = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        dsketch_rp = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        dresult_rp = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>());
        tau = std::unique_ptr<matrix::dense<index_type>>(new matrix::dense<index_type>());

        precond_mtx->generate({sampled_rows, size[1]});
        precond_mtx_internal->generate({sampled_rows, size[1]});
        dt->generate(size);
        mtx_rp->generate(size);
        dsketch_rp->generate({sampled_rows, size[0]});
        dresult_rp->generate({sampled_rows, size[1]});
        tau->generate_cpu({size[0], 1});
    }

    ~gaussian() {
        std::cout << "in ~gaussian\n";
    }

    void generate(double* runtime);

    void generate(matrix::dense<value_type>* mtx) {
        // this->set_type();
        auto num_rows = mtx->get_size()[0];
        auto num_cols = mtx->get_size()[1];
        index_type sampled_rows = (index_type)(sampling_coeff * num_cols);
        // Generates sketch matrix.
        sketch_mtx = std::shared_ptr<rls::matrix::dense<value_type>>(new rls::matrix::dense<value_type>({sampled_rows, num_rows}));
        sketch_mtx->generate();
        if (std::is_same<value_type, double>::value) {
            // auto status = 
            curandGenerateNormalDouble(
                this->info.rand_generator, (double*)sketch_mtx->get_values(), sampled_rows * num_rows, 0, 1);
        }
        else if (std::is_same<value_type, float>::value) {
            // auto status = 
            curandGenerateNormal(
                this->info.rand_generator, (float*)sketch_mtx->get_values(), sampled_rows * num_rows, 0, 1);
        }
        cudaDeviceSynchronize();
        auto size = mtx->get_size();
        precond_mtx = std::shared_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>({sampled_rows, size[1]}));
        std::cout << "precond_mtx->get_size()[0]: " << precond_mtx->get_size()[0] << ", precond_mtx->get_size()[1]: " << precond_mtx->get_size()[1] << '\n';
        precond_mtx_internal = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>({sampled_rows, size[1]}));
        dt = std::unique_ptr<matrix::dense<value_type>>(new matrix::dense<value_type>(size));
        mtx_rp = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>(size));
        dsketch_rp = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>({sampled_rows, size[0]}));
        dresult_rp = std::unique_ptr<matrix::dense<value_type_in>>(new matrix::dense<value_type_in>({sampled_rows, size[1]}));
        tau = std::unique_ptr<matrix::dense<index_type>>(new matrix::dense<index_type>({size[0], 1}));

        // std::cout << "sampled_rows: " << sampled_rows << ", size[1]: " << size[1] << '\n';
        std::cout << "allocating precond mtx\n";
        precond_mtx->generate();
        std::cout << "precond_mtx[0]: " << '\n';
        rls::io::print_mtx_gpu(1, 1, (double*)precond_mtx->get_values(), 1, this->info.queue);
        precond_mtx_internal->generate();
        dt->generate();
        mtx_rp->generate();
        dsketch_rp->generate();
        dresult_rp->generate();
        tau->generate_cpu();
    }

    void print();

    void allocate(dim2 size, index_type sampled_rows) {
        // sketch_mtx.generate(sampled_rows, size[0]);
        // precond_mtx.generate(size);
        // precond_mtx_internal.generate(size);
        // dt.generate(size);
        // mtx_rp.generate(size);
        // dsketch_rp.generate({sampled_rows, size[0]});
        // dresult_rp.generate({sampled_rows, size[1]});
        // tau.generate(size[0]);
    }

    gaussian<value_type_in, value_type, index_type>* get() {
        return this;
    }

    matrix::dense<value_type>* get_mtx() {
        return precond_mtx.get();
    }

    void apply(magma_trans_t trans, value_type* u_vector, index_type inc_u)
    {
        auto size = precond_mtx->get_size();
        blas::trsv(MagmaUpper, trans, MagmaNonUnit, size[1], precond_mtx->get_values(),
            size[0], u_vector, inc_u, this->info.queue);
    }

    void compute(matrix::dense<value_type>* dmtx)
    {
        value_type* hat_mtx = nullptr;
        double runtime;
        std::cout << "dmtx\n";
        rls::io::print_mtx_gpu(1, 1, (double*)dmtx->get_values(), 1, this->info.queue);
        compute_precond<value_type_in, value_type, index_type>(sketch_mtx->get_size()[0], sketch_mtx->get_size()[1],
                       sketch_mtx->get_values(), sketch_mtx->get_size()[0],
                       dmtx->get_size()[0], dmtx->get_size()[1], dmtx->get_values(),
                       dmtx->get_size()[0], precond_mtx->get_values(),
                       precond_mtx->get_size()[0], hat_mtx, this->info, &runtime);
        std::cout << "COMPUTE" << "\n";
        rls::io::print_mtx_gpu(1, 1, (double*)precond_mtx->get_values(), 1, this->info.queue);
    }

private:
    value_type sampling_coeff;
    std::shared_ptr<matrix::dense<value_type>> sketch_mtx;
    std::shared_ptr<matrix::dense<value_type>> precond_mtx;
    std::unique_ptr<matrix::dense<value_type_in>> precond_mtx_internal;
    std::unique_ptr<matrix::dense<value_type>> dt;
    std::unique_ptr<matrix::dense<value_type_in>> mtx_rp;
    std::unique_ptr<matrix::dense<value_type_in>> dsketch_rp;
    std::unique_ptr<matrix::dense<value_type_in>> dresult_rp;
    std::unique_ptr<matrix::dense<index_type>> tau;
};


template <typename value_type_internal, typename value_type,
          typename index_type>
void generate(index_type num_rows_sketch, index_type num_cols_sketch, value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx, value_type* dmtx, index_type ld_mtx,
              value_type* dr_factor, index_type ld_r_factor,
              value_type* hat_mtx, detail::magma_info& info);

template <typename value_type_internal, typename value_type,
          typename index_type>
void generate(index_type num_rows_sketch, index_type num_cols_sketch,
              value_type* dsketch, index_type ld_sketch,
              index_type num_rows_mtx, index_type num_cols_mtx,
              value_type* dmtx, index_type ld_mtx, value_type* dr_factor,
              index_type ld_r_factor, value_type* hat_mtx,
              detail::magma_info& info, double* runtime);


}  // namespace preconditioner
}  // namespace rls


#endif
