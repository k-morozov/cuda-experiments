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

add_executable(cuda_experiments src/main.cu src/gemm.cu)

target_link_libraries(cuda_experiments PRIVATE CUDA::cudart)
target_include_directories(cuda_experiments PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
