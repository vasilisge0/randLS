#ifndef RANDLS
#define RANDLS


#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"
#include <iostream>


#include "../utils/init_kernels.hpp"
#include "../utils/io.hpp"
#include "../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../core/memory/detail.hpp"
#include "../core/solver/lsqr.hpp"
#include "../cuda/solver/lsqr_kernels.cuh"


#endif
