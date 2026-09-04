cmake_minimum_required(VERSION 4.2)

project(cuda_experiments LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "Need GNU C++ compiler (g++), found: ${CMAKE_CXX_COMPILER_ID}")
endif()
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR "Need g++ >= 13.0, found: ${CMAKE_CXX_COMPILER_VERSION}")
endif()
message(STATUS "g++ ${CMAKE_CXX_COMPILER_VERSION} found: ${CMAKE_CXX_COMPILER}")

if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "Need NVIDIA CUDA compiler (nvcc), found: ${CMAKE_CUDA_COMPILER_ID}")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "Need nvcc >= 12.0, found: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()
message(STATUS "nvcc ${CMAKE_CUDA_COMPILER_VERSION} found: ${CMAKE_CUDA_COMPILER}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_library(gemm STATIC
    src/gemm_v1.cu
    src/gemm_v2.cu
    src/gemm_v3.cu
    src/gemm_v4.cu)
target_include_directories(gemm PUBLIC src)
target_link_libraries(gemm PUBLIC CUDA::cudart)

add_library(gemm_demo STATIC examples/gemm_demo.cpp)
target_include_directories(gemm_demo PUBLIC examples)
target_link_libraries(gemm_demo PUBLIC gemm)

add_executable(example_gemm_v1 examples/gemm_v1.cu)
target_link_libraries(example_gemm_v1 PRIVATE gemm_demo)

add_executable(example_gemm_v2 examples/gemm_v2.cu)
target_link_libraries(example_gemm_v2 PRIVATE gemm_demo)

add_executable(example_gemm_v3 examples/gemm_v3.cu)
target_link_libraries(example_gemm_v3 PRIVATE gemm_demo)

add_executable(example_gemm_v4 examples/gemm_v4.cu)
target_link_libraries(example_gemm_v4 PRIVATE gemm_demo)

add_library(gemm_square_demo STATIC examples/gemm_square_demo.cpp)
target_include_directories(gemm_square_demo PUBLIC examples)
target_link_libraries(gemm_square_demo PUBLIC gemm)

add_executable(example_gemm_v4_16x16 examples/gemm_v4_16x16.cu)
target_link_libraries(example_gemm_v4_16x16 PRIVATE gemm_square_demo)

add_executable(example_gemm_v4_32x32 examples/gemm_v4_32x32.cu)
target_link_libraries(example_gemm_v4_32x32 PRIVATE gemm_square_demo)

add_executable(example_gemm_v4_square examples/gemm_v4_square.cu)
target_link_libraries(example_gemm_v4_square PRIVATE gemm_square_demo)

add_library(gemm_unique_demo STATIC examples/gemm_unique_demo.cpp)
target_include_directories(gemm_unique_demo PUBLIC examples)
target_link_libraries(gemm_unique_demo PUBLIC gemm)

add_executable(example_gemm_v4_8x8_unique examples/gemm_v4_8x8_unique.cu)
target_link_libraries(example_gemm_v4_8x8_unique PRIVATE gemm_unique_demo)

add_executable(example_gemm_v4_4x4_unique examples/gemm_v4_4x4_unique.cu)
target_link_libraries(example_gemm_v4_4x4_unique PRIVATE gemm_unique_demo)

option(CUDA_EXPERIMENTS_BUILD_TESTS "Build tests" ON)
if(CUDA_EXPERIMENTS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
