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

}
}
}
