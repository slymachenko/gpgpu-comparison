#include "CUDAkernel.cuh"

template <typename T>
__global__ void matrixCompute(
    const T *a,
    const T *b,
    const T *c,
    const T *d,
    const T *e,
    T *result,
    int width,
    int height)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < 100000; k++)
    {
        if (i < height && j < width)
        {
            int index = i * width + j;
            result[index] = a[index] + b[index] * c[index] - d[index] * e[index];
        }
    }
}

template __global__ void matrixCompute<float>(const float *a, const float *b, const float *c, const float *d, const float *e, float *result, int width, int height);
template __global__ void matrixCompute<double>(const double *a, const double *b, const double *c, const double *d, const double *e, double *result, int width, int height);
