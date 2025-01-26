#define NOMINMAX

#include <iostream>
#include <chrono>
#include <cmath>
#include "Matrix.h"
#include "ParallelCPU.h"
#include "OpenCLComputation.h"
#include "CUDAComputation.h"

const float EPSILON = 1e-4;

template <typename T>
bool compareResults(const Matrix<T> &ref, const Matrix<T> &test, const std::string &methodName)
{
    bool allMatch = true;
    for (int i = 0; i < ref.data.size(); i++)
    {
        if (std::fabs(ref.data[i] - test.data[i]) > EPSILON)
        {
            std::cerr << "Mismatch at index " << i << ": "
                      << "Expected " << ref.data[i] << ", but got " << test.data[i]
                      << " in " << methodName << std::endl;
            allMatch = false;
            break;
        }
    }
    return allMatch;
}

template <typename T>
void validateResults(const Matrix<T> &resultParallel,
                     const Matrix<T> &resultOpenCL,
                     const Matrix<T> &resultCUDA)
{
    bool openCLMatch = compareResults(resultParallel, resultOpenCL, "OpenCL");
    bool cudaMatch = compareResults(resultParallel, resultCUDA, "CUDA");

    if (openCLMatch && cudaMatch)
    {
        std::cout << "All results match!" << std::endl;
    }
    else
    {
        if (!openCLMatch)
            std::cerr << "OpenCL computation differs!" << std::endl;
        if (!cudaMatch)
            std::cerr << "CUDA computation differs!" << std::endl;
    }
}

template <typename T>
std::tuple<std::chrono::duration<double, std::milli>, std::chrono::duration<double, std::milli>> test(Computation<T> &computation)
{
    try
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto elapsedComputation = computation.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsedTotal = end - start;
        return std::make_tuple(elapsedTotal, elapsedComputation);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during computation: " << e.what() << std::endl;
        return std::make_tuple(std::chrono::duration<double>(0), std::chrono::duration<double>(0));
    }
}

int getValidIntInput(const std::string &prompt, int min, int max)
{
    int value;
    while (true)
    {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail() || value < min || value > max)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number between " << min << " and " << max << "." << std::endl;
        }
        else
        {
            break;
        }
    }
    return value;
}

template <typename T>
void compute(int width, int height)
{
    Matrix<T> a(width, height), b(width, height), c(width, height), d(width, height), e(width, height);
    Matrix<T> resultParallel(width, height);
    Matrix<T> resultOpenCL(width, height);
    Matrix<T> resultCUDA(width, height);
    ParallelCPU<T> pc(a, b, c, d, e, resultParallel);
    OpenCLComputation<T> ocl(a, b, c, d, e, resultOpenCL);
    CUDAComputation<T> cuda(a, b, c, d, e, resultCUDA);
    std::chrono::duration<double, std::milli> elapsedParallelTotal, elapsedParallelComp;
    std::chrono::duration<double, std::milli> elapsedOpenCLTotal, elapsedOpenCLComp;
    std::chrono::duration<double, std::milli> elapsedCUDATotal, elapsedCUDAComp;

    // OPENCL PLATFORM AND DEVICE SELECTION
    auto platforms = OpenCLComputation<T>::getAvailablePlatforms();
    std::cout << std::endl;
    OpenCLComputation<T>::printPlatforms(platforms);

    int platformIndex = 0;
    if (platforms.size() > 1)
    {
        platformIndex = getValidIntInput("Select OpenCL platform index: ", 0, platforms.size() - 1);
    }

    auto devices = OpenCLComputation<T>::getAvailableDevices(platforms[platformIndex]);
    std::cout << std::endl;
    OpenCLComputation<T>::printDevices(devices);
    std::cout << std::endl;

    int deviceIndex = 0;
    if (devices.size() > 1)
    {
        deviceIndex = getValidIntInput("Select OpenCL device index: ", 0, devices.size() - 1);
    }

    ocl.init(platformIndex, deviceIndex);

    // CUDA PLATFORM AND DEVICE SELECTION
    auto cudaDevices = CUDAComputation<T>::getAvailableDevices();
    std::cout << std::endl;
    CUDAComputation<T>::printDevices(cudaDevices);
    std::cout << std::endl;

    int cudaDeviceIndex = 0;
    if (cudaDevices.size() > 1)
    {
        cudaDeviceIndex = getValidIntInput("Select CUDA device index: ", 0, cudaDevices.size() - 1);
    }
    cuda.init(cudaDeviceIndex);

    // CALCULATIONS
    std::cout << "\nParallel Computation..." << std::endl;
    std::tie(elapsedParallelTotal, elapsedParallelComp) = test(pc);

    std::cout << "\nOpenCL Computation..." << std::endl;
    std::tie(elapsedOpenCLTotal, elapsedOpenCLComp) = test(ocl);

    std::cout << "\nCUDA Computation..." << std::endl;
    std::tie(elapsedCUDATotal, elapsedCUDAComp) = test(cuda);

    std::cout << "\nValidation..." << std::endl;
    validateResults(resultParallel, resultOpenCL, resultCUDA);

    // COMPARISON
    std::cout << "\nComparison:" << std::endl;
    std::cout << "Parallel: \t" << elapsedParallelTotal.count() << " ms" << std::endl;
    std::cout << "OpenCL: \t" << elapsedOpenCLTotal.count() << " ms" << std::endl;
    std::cout << "CUDA:   \t" << elapsedCUDATotal.count() << " ms" << std::endl;

    // PRINT MATRICES
    // a.print("A");
    // b.print("B");
    // c.print("C");
    // d.print("D");
    // e.print("E");
    // resultParallel.print("Parallel Result");
}

int main()
{
    int width, height, isDouble;

    isDouble = getValidIntInput("Enter matrix type (0: float; 1: double): ", 0, 1);
    width = getValidIntInput("Enter matrix width (1 to 10000): ", 1, 10000);
    height = getValidIntInput("Enter matrix height (1 to 10000): ", 1, 10000);

    if (isDouble)
        compute<double>(width, height);
    else
        compute<float>(width, height);

    return 0;
}
