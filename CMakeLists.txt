cmake_minimum_required(VERSION 3.17)
project(ComputationComparison LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(include)
include_directories($ENV{CUDA_PATH}/include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(KERNEL_PATH "${CMAKE_SOURCE_DIR}/src/OpenCLkernel.cl")
add_definitions(-DKERNEL_PATH="${KERNEL_PATH}")

set(SOURCES
    src/main.cpp
    src/Matrix.cpp
    src/ParallelCPU.cpp
    src/OpenCLComputation.cpp
    src/CUDAComputation.cu
    src/CUDAkernel.cu
)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_directories($ENV{CUDA_PATH}/lib/x64)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCL_LIBRARIES} ${CUDA_LIBRARIES})