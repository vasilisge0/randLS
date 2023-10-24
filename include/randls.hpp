#ifndef RANDLS
#define RANDLS


#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "magma_lapack.h"
#include "magma_v2.h"


#include "../utils/io.hpp"
#include "../cuda/preconditioner/preconditioner_kernels.cuh"
#include "../core/preconditioner/preconditioner.hpp"
#include "../core/preconditioner/sketchqr.hpp"
#include "../core/preconditioner/generalized_split.hpp"
#include "../core/sketch/gaussian.hpp"
#include "../core/sketch/countsketch.hpp"
#include "../core/sketch/sketch.hpp"
#include "../core/memory/magma_context.hpp"
#include "../core/memory/memory.hpp"
#include "../core/solver/lsqr.hpp"
#include "../core/solver/fgmres.hpp"
#include "../core/solver/ir.hpp"
#include "../cuda/solver/lsqr_kernels.cuh"
#include "../core/matrix/dense/dense.hpp"
#include "../core/matrix/sparse/sparse.hpp"


#endif
