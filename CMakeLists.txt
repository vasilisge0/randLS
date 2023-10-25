cmake_minimum_required(VERSION 3.13.11)
project(mix_n_pick LANGUAGES CXX C CUDA VERSION 1.0)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "-Wall -std=c++14")

add_library(mix_n_pick SHARED
            core/memory/magma_context.cpp
            core/memory/memory.cpp
            core/matrix/mtx.cpp
            core/matrix/mtxop.cpp
            core/matrix/dense/dense.cpp
            core/matrix/sparse/sparse.cpp
            core/solver/lsqr.cpp
            core/solver/fgmres.cpp
            core/sketch/gaussian.cpp
            core/preconditioner/preconditioner.cpp
            core/preconditioner/sketchqr.cpp
            core/preconditioner/generalized_split.cpp
            cuda/sketch/countsketch.cu
            cuda/solver/lsqr_kernels.cu
            cuda/preconditioner/preconditioner_kernels.cu
            cuda/sketch/gaussiansketch.cu
            core/sketch/countsketch.cpp
            utils/convert.cpp
            utils/io.cpp
)

target_include_directories(mix_n_pick PUBLIC
    .
    ../
    include/
    ${HOME}/ginkgo_clone_install/include
    core/memory/
    core/matrix/
    core/matrix/dense/
    core/matrix/sparse/
    core/preconditioner/
    core/solver/
    core/sketch
    ${MAGMA_INC}
    ${CUDA_INC}
    ${GINKGO_INC}
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
    #-std=c++14 ${MAGMA_LIB}/libmagma.so ${GINKGO_LIB}/libginkgo.so
    -std=c++14 ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lcudart -lcurand -lcudadevrt ${GINKGO_LIB}/libginkgo.so
)
enable_language(CUDA)

# I/O

add_library(mmio SHARED utils/mmio.c)
target_include_directories(mmio PUBLIC
    .
    ../
    core
    utils
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
    ${GINKGO_INC}
)
#target_link_libraries(mix_n_pick
#    -std=c++14 ${MAGMA_LIB}/libmagma.so ${GINKGO_LIB}/libginkgo.so
#    #-std=c++14 ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lcudart -lcurand -lcudadevrt ${GINKGO_LIB}/libginkgo.so
#)

# run_lsqr

add_executable(run_lsqr runners/run_lsqr.cpp)
add_dependencies(run_lsqr mmio)
add_dependencies(run_lsqr mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_lsqr PUBLIC
    .
    ../
    include/
    core/sketch
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
    ${GINKGO_INC}
)
set_target_properties(run_lsqr PROPERTIES 
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_lsqr
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
)

# run_lsqr_sparse_all

add_executable(run_lsqr_sparse_all runners/run_lsqr_sparse_all.cpp)
add_dependencies(run_lsqr_sparse_all mmio)
add_dependencies(run_lsqr_sparse_all mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_lsqr_sparse_all PUBLIC
    .
    ../
    include/
    core/sketch
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
    ${GINKGO_INC}
)
set_target_properties(run_lsqr_sparse_all PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_lsqr_sparse_all
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
)

# run_fgmres_sparse_all

add_executable(run_fgmres_sparse_all runners/run_fgmres_sparse_all.cpp)
add_dependencies(run_fgmres_sparse_all mmio)
add_dependencies(run_fgmres_sparse_all mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_fgmres_sparse_all PUBLIC
    .
    ../
    include/
    core/sketch
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
    ${GINKGO_INC}
)
set_target_properties(run_fgmres_sparse_all PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_fgmres_sparse_all
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
)


## run_lsqr_all
#
#add_executable(run_lsqr_all runners/run_lsqr_all.cpp)
#add_dependencies(run_lsqr_all mmio)
#add_dependencies(run_lsqr_all mix_n_pick)
## find_library(MAGMA_LIB libmagma.so)
#target_include_directories(run_lsqr_all PUBLIC
#    .
#    ../
#    include/
#    core/sketch
#    ${MAGMA_INC}
#    ${CUDA_INC}
#    ${RANDLS_INC}
#    ${GINKGO_INC}
#)
#set_target_properties(run_lsqr_all PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS NO
#    LINKER_LANGUAGE CUDA)
#target_link_libraries(run_lsqr_all
#   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
#)

## run_fgmres
#
#add_executable(run_fgmres runners/run_fgmres.cpp)
#add_dependencies(run_fgmres mmio)
#add_dependencies(run_fgmres mix_n_pick)
## find_library(MAGMA_LIB libmagma.so)
#target_include_directories(run_fgmres PUBLIC
#    .
#    ../
#    include/
#    ${MAGMA_INC}
#    ${CUDA_INC}
#    ${RANDLS_INC}
#    ${GINKGO_INC}
#)
#set_target_properties(run_fgmres PROPERTIES 
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS NO
#    LINKER_LANGUAGE CUDA)
#target_link_libraries(run_fgmres
#   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lmix_n_pick -lcublas -lcusparse -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so
#)

## run_fgmres_example0
#
#add_executable(run_fgmres_example0 runners/run_fgmres_example0.cpp)
#add_dependencies(run_fgmres_example0 mmio)
#add_dependencies(run_fgmres_example0 mix_n_pick)
## find_library(MAGMA_LIB libmagma.so)
#target_include_directories(run_fgmres_example0 PUBLIC
#    .
#    ../
#    include/
#    ${MAGMA_INC}
#    ${CUDA_INC}
#    ${RANDLS_INC}
#    ${GINKGO_INC}
#    ${HOME}/ginkgo_clone_install/include
#)
#set_target_properties(run_fgmres_example0 PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS NO
#    LINKER_LANGUAGE CUDA)
#target_link_libraries(run_fgmres_example0
#   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lmix_n_pick -lcublas -lcusparse -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so
#)


# run_fgmres_example_large0

#add_executable(run_fgmres_example_large0 runners/run_fgmres_example_large0.cpp)
#add_dependencies(run_fgmres_example_large0 mmio)
#add_dependencies(run_fgmres_example_large0 mix_n_pick)
## find_library(MAGMA_LIB libmagma.so)
#target_include_directories(run_fgmres_example_large0 PUBLIC
#    .
#    ../
#    include/
#    ${MAGMA_INC}
#    ${CUDA_INC}
#    ${RANDLS_INC}
#    ${GINKGO_INC}
#    ${HOME}/ginkgo_clone_install/include
#)
#set_target_properties(run_fgmres_example_large0 PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS NO
#    LINKER_LANGUAGE CUDA)
#target_link_libraries(run_fgmres_example_large0
#   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lmix_n_pick -lcublas -lcusparse -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
#)

# run_fgmres_sparse_small

add_executable(run_fgmres_sparse_small runners/run_fgmres_sparse_small.cpp)
add_dependencies(run_fgmres_sparse_small mmio)
add_dependencies(run_fgmres_sparse_small mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_fgmres_sparse_small PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
    ${GINKGO_INC}
    ${HOME}/ginkgo_clone_install/include
)
set_target_properties(run_fgmres_sparse_small PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_fgmres_sparse_small
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lmix_n_pick -lcublas -lcusparse -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
)

# run_fgmres_dense

add_executable(run_fgmres_dense runners/run_fgmres_dense.cpp)
add_dependencies(run_fgmres_dense mmio)
add_dependencies(run_fgmres_dense mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_fgmres_dense PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
    ${GINKGO_INC}
    ${HOME}/ginkgo_clone_install/include
)
set_target_properties(run_fgmres_dense PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_fgmres_dense
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lmix_n_pick -lcublas -lcusparse -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
)

# run_lsqr_sparse

add_executable(run_lsqr_sparse runners/run_lsqr_sparse.cpp)
add_dependencies(run_lsqr_sparse mmio)
add_dependencies(run_lsqr_sparse mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
target_include_directories(run_lsqr_sparse PUBLIC
    .
    ../
    include/
    ${MAGMA_INC}
    ${CUDA_INC}
    ${RANDLS_INC}
    ${GINKGO_INC}
    ${HOME}/ginkgo_clone_install/include
)
set_target_properties(run_lsqr_sparse PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_lsqr_sparse
   -std=c++14 -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lmix_n_pick -lcublas -lcusparse -lmmio -lcudart -lcurand ${GINKGO_LIB}/libginkgo.so ${GINKGO_LIB}/libginkgo_reference.so ${GINKGO_LIB}/libginkgo_omp.so  ${GINKGO_LIB}/libginkgo_cuda.so ${GINKGO_LIB}/libginkgo_hip.so ${GINKGO_LIB}/libginkgo_dpcpp.so ${GINKGO_LIB}/libginkgo_device.so
)
