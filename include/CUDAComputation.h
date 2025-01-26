#pragma once
#include "Matrix.h"
#include "Computation.h"
#include <cuda_runtime.h>

template <typename T>
class CUDAComputation : public Computation<T>
{
public:
    CUDAComputation(const Matrix<T> &a, const Matrix<T> &b, const Matrix<T> &c, const Matrix<T> &d, const Matrix<T> &e, Matrix<T> &result);
    void init(int device_ind);
    std::chrono::duration<double, std::milli> run() override;

    static std::vector<cudaDeviceProp> getAvailableDevices();
    static void printDevices(const std::vector<cudaDeviceProp> &devices);

private:
    void selectDevice(int index);
    void initBuffers();
    void freeBuffers();

    size_t bufferSize;
    T *d_a;
    T *d_b;
    T *d_c;
    T *d_d;
    T *d_e;
    T *d_result;
    int deviceIndex;
};
