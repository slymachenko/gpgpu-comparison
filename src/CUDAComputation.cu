#include "CUDAComputation.h"
#include "CUDAkernel.cuh"
#include <iostream>
#include <chrono>

template <typename T>
CUDAComputation<T>::CUDAComputation(const Matrix<T> &a, const Matrix<T> &b, const Matrix<T> &c, const Matrix<T> &d, const Matrix<T> &e, Matrix<T> &result)
    : Computation<T>(a, b, c, d, e, result)
{
    bufferSize = this->width * this->height * sizeof(T);
}

template <typename T>
void CUDAComputation<T>::init(int device_ind)
{
    selectDevice(device_ind);
}

template <typename T>
std::vector<cudaDeviceProp> CUDAComputation<T>::getAvailableDevices()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::vector<cudaDeviceProp> devices(deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaGetDeviceProperties(&devices[i], i);
    }
    return devices;
}

template <typename T>
void CUDAComputation<T>::printDevices(const std::vector<cudaDeviceProp> &devices)
{
    std::cout << "Available CUDA devices:" << std::endl;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        std::cout << "Device " << i << ": " << devices[i].name << std::endl;
    }
}

template <typename T>
void CUDAComputation<T>::selectDevice(int index)
{
    std::vector<cudaDeviceProp> devices = getAvailableDevices();
    if (index < 0 || index >= devices.size())
    {
        throw std::runtime_error("Invalid device index.");
    }
    deviceIndex = index;
    cudaSetDevice(deviceIndex);
    std::cout << "Using CUDA device: " << devices[deviceIndex].name << std::endl;
}

template <typename T>
void CUDAComputation<T>::initBuffers()
{
    cudaMalloc(&d_a, bufferSize);
    cudaMalloc(&d_b, bufferSize);
    cudaMalloc(&d_c, bufferSize);
    cudaMalloc(&d_d, bufferSize);
    cudaMalloc(&d_e, bufferSize);
    cudaMalloc(&d_result, bufferSize);
}

template <typename T>
void CUDAComputation<T>::freeBuffers()
{
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
    cudaFree(d_result);
}

template <typename T>
std::chrono::duration<double, std::milli> CUDAComputation<T>::run()
{
    initBuffers();

    cudaMemcpy(d_a, this->a.data.data(), bufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, this->b.data.data(), bufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, this->c.data.data(), bufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, this->d.data.data(), bufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, this->e.data.data(), bufferSize, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((this->width + blockSize.x - 1) / blockSize.x, (this->height + blockSize.y - 1) / blockSize.y);

    auto computeStart = std::chrono::high_resolution_clock::now();
    matrixCompute<<<gridSize, blockSize>>>(d_a, d_b, d_c, d_d, d_e, d_result, this->width, this->height);
    cudaDeviceSynchronize();
    auto computeEnd = std::chrono::high_resolution_clock::now();

    cudaMemcpy(this->result.data.data(), d_result, bufferSize, cudaMemcpyDeviceToHost);

    freeBuffers();

    return computeEnd - computeStart;
}

template class CUDAComputation<float>;
template class CUDAComputation<double>;
