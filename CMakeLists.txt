cmake_minimum_required(VERSION 3.13.11)
project(mix_n_pick LANGUAGES CXX C CUDA VERSION 1.0)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "-Wall -std=c++14")

add_library(mix_n_pick SHARED
            utils/convert.cpp
            core/solver/lsqr.cpp
            core/solver/fgmres.cpp
            core/solver/solver.cpp
            cuda/solver/lsqr_kernels.cu
            cuda/preconditioner/preconditioner_kernels.cu
            utils/init_kernels.cpp
            core/blas/blas.cpp
            core/memory/memory.cpp
            core/preconditioner/gaussian.cpp
            core/preconditioner/preconditioner.cpp
)
target_include_directories(mix_n_pick PUBLIC
    .
    ../
    include/
    core/preconditioner/
    core/solver/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
)
# set those here or while configuring cmake 
set_property(TARGET mix_n_pick
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET mix_n_pick
             PROPERTY CUDA_STANDARD 14)
#set(MAGMA_INC )
#set(CUDA_INC )
# set(MAGMA_LIB )
# find_library(MAGMA_LIB libmagma.so)
# target_include_directories(mix_n_pick PUBLIC
#     .
#     ../
#     core/
#     utils/
#     ../test/
#     ../cuda/
#     include
#     ${MAGMA_INC}
#     ${RANDLS_INC}
# )
message("MAGMA_LIB: " ${MAGMA_LIB})
message("cudainc: " ${CUDA_INC})
set_target_properties(mix_n_pick PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(mix_n_pick
    -std=c++14 ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lcudart -lcurand
)
enable_language(CUDA)

# I/O

add_library(mmio SHARED utils/mmio.c utils/io.cpp)
target_include_directories(mmio PUBLIC
    .
    ../
    core
    utils
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
)

# run_lsqr

add_executable(run_lsqr runners/run_lsqr.cpp)
add_dependencies(run_lsqr mmio)
add_dependencies(run_lsqr mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_lsqr PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
)
set_target_properties(run_lsqr PROPERTIES 
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_lsqr
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
)

# run_fgmres

add_executable(run_fgmres runners/run_fgmres.cpp)
add_dependencies(run_fgmres mmio)
add_dependencies(run_fgmres mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_fgmres PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
)
set_target_properties(run_fgmres PROPERTIES 
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_fgmres
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
)

# run_ir

add_executable(run_ir runners/run_ir.cpp)
add_dependencies(run_ir mmio)
add_dependencies(run_ir mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_ir PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
)
set_target_properties(run_ir PROPERTIES 
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_ir
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
)
