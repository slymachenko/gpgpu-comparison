#include "OpenCLComputation.h"
#include <fstream>
#include <sstream>
#include <iostream>

template <typename T>
OpenCLComputation<T>::OpenCLComputation(const Matrix<T> &a, const Matrix<T> &b, const Matrix<T> &c, const Matrix<T> &d, const Matrix<T> &e, Matrix<T> &result)
    : Computation(a, b, c, d, e, result) {}

template <typename T>
void OpenCLComputation<T>::init(int platform_ind, int device_ind, const std::string &krnlPath)
{
    kernelPath = krnlPath;
    selectPlatform(platform_ind);
    selectDevice(device_ind);
}

template <typename T>
std::vector<cl::Platform> OpenCLComputation<T>::getAvailablePlatforms()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    return platforms;
}

template <typename T>
void OpenCLComputation<T>::printPlatforms(const std::vector<cl::Platform> &platforms)
{
    std::cout << "Available OpenCL platforms:" << std::endl;
    for (size_t i = 0; i < platforms.size(); ++i)
    {
        std::cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
    }
}

template <typename T>
void OpenCLComputation<T>::selectPlatform(int index)
{
    std::vector<cl::Platform> platforms = getAvailablePlatforms();
    if (index < 0 || index >= platforms.size())
    {
        throw std::runtime_error("Invalid platform index.");
    }
    platform = platforms[index];
    std::cout << "Using OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
}

template <typename T>
std::vector<cl::Device> OpenCLComputation<T>::getAvailableDevices(const cl::Platform &platform)
{
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    return devices;
}

template <typename T>
void OpenCLComputation<T>::printDevices(const std::vector<cl::Device> &devices)
{
    std::cout << "Available OpenCL devices:" << std::endl;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        std::cout << "Device " << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    }
}

template <typename T>
void OpenCLComputation<T>::selectDevice(int index)
{
    std::vector<cl::Device> devices = getAvailableDevices(platform);
    if (index < 0 || index >= devices.size())
    {
        throw std::runtime_error("Invalid device index.");
    }
    device = devices[index];
    std::cout << "Using OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
}

template <typename T>
void OpenCLComputation<T>::initBuffers()
{
    bufferSize = static_cast<cl::size_type>(this->width * this->height * sizeof(T));
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);
    bufferA = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, (void *)this->a.data.data());
    bufferB = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, (void *)this->b.data.data());
    bufferC = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, (void *)this->c.data.data());
    bufferD = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, (void *)this->d.data.data());
    bufferE = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, (void *)this->e.data.data());
    bufferResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferSize);
}

template <typename T>
std::string OpenCLComputation<T>::loadKernel(const std::string &kernelPath)
{
    std::ifstream file(kernelPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open kernel file: " + kernelPath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

template <typename T>
void OpenCLComputation<T>::initKernel(const std::string &kernelPath)
{
    std::string kernelSource = loadKernel(kernelPath);

    if (std::is_same<T, double>::value)
    {
        kernelSource = "#define USE_DOUBLE\n" + kernelSource;
    }

    cl::Program program(context, kernelSource);
    program.build(device);
    kernel = cl::Kernel(program, "matrixCompute");
}

template <typename T>
std::chrono::duration<double, std::milli> OpenCLComputation<T>::run()
{
    initBuffers();
    initKernel(kernelPath);

    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, bufferD);
    kernel.setArg(4, bufferE);
    kernel.setArg(5, bufferResult);
    kernel.setArg(6, this->width);
    kernel.setArg(7, this->height);

    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, bufferSize, this->a.data.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, bufferSize, this->b.data.data());
    queue.enqueueWriteBuffer(bufferC, CL_TRUE, 0, bufferSize, this->c.data.data());
    queue.enqueueWriteBuffer(bufferD, CL_TRUE, 0, bufferSize, this->d.data.data());
    queue.enqueueWriteBuffer(bufferE, CL_TRUE, 0, bufferSize, this->e.data.data());

    auto computeStart = std::chrono::high_resolution_clock::now();
    cl::NDRange global(this->width, this->height);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();
    auto computeEnd = std::chrono::high_resolution_clock::now();

    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, bufferSize, this->result.data.data());

    return computeEnd - computeStart;
}

template class OpenCLComputation<float>;
template class OpenCLComputation<double>;
