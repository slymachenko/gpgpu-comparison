#pragma once
#include <cuda_runtime.h>

template <typename T>
__global__ void matrixCompute(const T* a, const T* b, const T* c, const T* d, const T* e, T* result, int width, int height);
