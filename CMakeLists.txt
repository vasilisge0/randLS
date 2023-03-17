cmake_minimum_required(VERSION 3.13.11)
project(randls LANGUAGES CXX C CUDA VERSION 1.0)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11")

# build .so

add_library(randls SHARED
            cuda/preconditioner/preconditioner_kernels.cu
            cuda/solver/lsqr_kernels.cu
            core/solver/lsqr.cpp
            utils/init_kernels.cpp
            core/blas/blas.cpp
            core/memory/detail.cpp
            core/memory/memory.cpp
            core/preconditioner/gaussian.cpp
)
target_include_directories(randls PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
)

# set those here or while configuring cmake 
set_property(TARGET randls
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET randls
             PROPERTY CUDA_STANDARD 11)
set(MAGMA_INC )
set(CUDA_INC )
set_target_properties(randls PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(randls
    -std=c++11 ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lcudart -lcurand
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
    ${RBPIK_INC}
)

# compile runner (run_lsqr)

add_executable(run_lsqr runners/run_lsqr.cpp)
add_dependencies(run_lsqr mmio)
add_dependencies(run_lsqr randls)
target_include_directories(run_lsqr PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RBPIK_INC}
)
set_target_properties(run_lsqr PROPERTIES 
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)

target_link_libraries(run_lsqr
    -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lrandls -lmmio -lcudart -lcurand
)
