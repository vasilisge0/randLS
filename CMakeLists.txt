cmake_minimum_required(VERSION 3.13.11)
project(mix_n_pick LANGUAGES CXX C CUDA VERSION 1.0)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11")

add_library(mix_n_pick SHARED
            core/solver/lsqr.cpp
            core/solver/solver.cpp
            cuda/solver/lsqr_kernels.cu
            cuda/preconditioner/preconditioner_kernels.cu
            utils/init_kernels.cpp
            core/blas/blas.cpp
            core/memory/detail.cpp
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
    ${RBPIK_INC}
)
# set those here or while configuring cmake 
set_property(TARGET mix_n_pick
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET mix_n_pick
             PROPERTY CUDA_STANDARD 11)
set(MAGMA_INC )
set(CUDA_INC )
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
#     ${RBPIK_INC}
# )
message("MAGMA_LIB: " ${MAGMA_LIB})
set_target_properties(mix_n_pick PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(mix_n_pick
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
    ${RBPIK_INC}
)
set_target_properties(run_lsqr PROPERTIES 
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CUDA)
target_link_libraries(run_lsqr
    -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
)

# # run_lsqr_single

# add_executable(run_lsqr_single runners/run_lsqr_single.cpp)
# add_dependencies(run_lsqr_single mmio)
# add_dependencies(run_lsqr_single mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
# target_include_directories(run_lsqr_single PUBLIC
#     .
#     ../
#     include/
#     ${MAGMA_INC}
#     ${CUDA_INC}
#     ${RBPIK_INC}
#     /software/all/devel/cuda/11.0/include
# )
# set_target_properties(run_lsqr_single PROPERTIES LINKER_LANGUAGE CUDA)
# target_link_libraries(run_lsqr_single
#     -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
# )

# # run_lsqr_half

# add_executable(run_lsqr_half test/run_lsqr_half.cpp)
# add_dependencies(run_lsqr_half mmio)
# add_dependencies(run_lsqr_half mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
# target_include_directories(run_lsqr_half PUBLIC
#     .
#     ../
#     core
#     utils
#     test
#     ${MAGMA_INC}
#     ${CUDA_INC}
#     ${RBPIK_INC}
#     /software/all/devel/cuda/11.0/include
# )
# set_target_properties(run_lsqr_half PROPERTIES LINKER_LANGUAGE CUDA)
# target_link_libraries(run_lsqr_half
#     -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
# )

# # run_lsqr_stats

# add_executable(run_lsqr_stats test/run_lsqr_stats.cpp)
# add_dependencies(run_lsqr_stats mmio)
# add_dependencies(run_lsqr_stats mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
# target_include_directories(run_lsqr_stats PUBLIC
#     .
#     ../
#     core
#     utils
#     test
#     ${MAGMA_INC}
#     ${CUDA_INC}
#     ${RBPIK_INC}
#     /software/all/devel/cuda/11.0/include
# )
# set_target_properties(run_lsqr_stats PROPERTIES LINKER_LANGUAGE CUDA)
# target_link_libraries(run_lsqr_stats
#     -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
# )

# # run_lsqr_single_stats

# add_executable(run_lsqr_single_stats test/run_lsqr_single_stats.cpp)
# add_dependencies(run_lsqr_single_stats mmio)
# add_dependencies(run_lsqr_single_stats mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
# target_include_directories(run_lsqr_single_stats PUBLIC
#     .
#     ../
#     core
#     utils
#     test
#     ${MAGMA_INC}
#     ${CUDA_INC}
#     ${RBPIK_INC}
#     /software/all/devel/cuda/11.0/include
# )
# set_target_properties(run_lsqr_single_stats PROPERTIES LINKER_LANGUAGE CUDA)
# target_link_libraries(run_lsqr_single_stats
#     -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
# )

# # run_lsqr_tf32_stats

# add_executable(run_lsqr_tf32_stats test/run_lsqr_tf32_stats.cpp)
# add_dependencies(run_lsqr_tf32_stats mmio)
# add_dependencies(run_lsqr_tf32_stats mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
# target_include_directories(run_lsqr_tf32_stats PUBLIC
#     .
#     ../
#     core
#     utils
#     test
#     ${MAGMA_INC}
#     ${CUDA_INC}
#     ${RBPIK_INC}
#     /software/all/devel/cuda/11.0/include
# )
# set_target_properties(run_lsqr_tf32_stats PROPERTIES LINKER_LANGUAGE CUDA)
# target_link_libraries(run_lsqr_tf32_stats
#     -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
# )

# # run_lsqr_half_stats

# add_executable(run_lsqr_half_stats test/run_lsqr_half_stats.cpp)
# add_dependencies(run_lsqr_half_stats mmio)
# add_dependencies(run_lsqr_half_stats mix_n_pick)
# find_library(MAGMA_LIB libmagma.so)
# target_include_directories(run_lsqr_half_stats PUBLIC
#     .
#     ../
#     core
#     utils
#     test
#     ${MAGMA_INC}
#     ${CUDA_INC}
#     ${RBPIK_INC}
#     /software/all/devel/cuda/11.0/include
# )
# set_target_properties(run_lsqr_half_stats PROPERTIES LINKER_LANGUAGE CUDA)
# target_link_libraries(run_lsqr_half_stats
#     -L${PROJECT_BINARY_DIR} ${MAGMA_LIB}/libmagma.so -lcublas -lcusparse -lmix_n_pick -lmmio -lcudart -lcurand
# )
