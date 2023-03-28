#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../blas/blas.hpp"
#include "../dense/dense.hpp"
#include "../memory/memory.hpp"
#include "../preconditioner/preconditioner.hpp"
#include "base_types.hpp"
#include "lsqr.hpp"
#include "solver.hpp"

#include "../../utils/io.hpp"
#include "../../cuda/solver/lsqr_kernels.cuh"


namespace rls {
namespace solver {
namespace {

} // end of anonymous namespace




} // end of solver namespace
} // end of rls namespace