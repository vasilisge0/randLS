#ifndef GAUSSIANSKETCH_HPP
#define GAUSSIANSKETCH_HPP


#include "../../core/matrix/dense/dense.hpp"
#include "../../include/magma_context.hpp"


void gaussian_sketch_impl(matrix::Dense<CUDA, double> mtx);

void gaussian_sketch_impl(matrix::Dense<CUDA, float>);


#endif
