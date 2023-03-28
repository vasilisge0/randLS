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
#include "../core/preconditioner/preconditioner.hpp"
#include "../core/preconditioner/gaussian.hpp"
#include "../core/memory/magma_context.hpp"
#include "../core/solver/lsqr.hpp"
#include "../core/solver/fgmres.hpp"
#include "../cuda/solver/lsqr_kernels.cuh"
#include "../core/dense/dense.hpp"


#endif
