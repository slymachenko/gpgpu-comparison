#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include "Matrix.h"
#include "Computation.h"
#include <CL/cl.hpp>

template <typename T>
class OpenCLComputation : public Computation<T>
{
public:
    OpenCLComputation(const Matrix<T> &a, const Matrix<T> &b, const Matrix<T> &c, const Matrix<T> &d, const Matrix<T> &e, Matrix<T> &result);
    void init(int platform_ind, int device_ind, const std::string &kernelPath = KERNEL_PATH);
    std::chrono::duration<double, std::milli> run() override;

    static std::vector<cl::Platform> getAvailablePlatforms();
    static void printPlatforms(const std::vector<cl::Platform> &platforms);

    static std::vector<cl::Device> getAvailableDevices(const cl::Platform &platform);
    static void printDevices(const std::vector<cl::Device> &devices);

private:
    void initBuffers();
    std::string loadKernel(const std::string &kernelPath);
    void initKernel(const std::string &kernelPath);
    void selectPlatform(int index);
    void selectDevice(int index);

    cl::Platform platform;
    cl::Device device;
    cl::size_type bufferSize;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Buffer bufferA, bufferB, bufferC, bufferD, bufferE, bufferResult;
    cl::Kernel kernel;
    std::string kernelPath;
};
