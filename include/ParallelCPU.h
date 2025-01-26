#pragma once
#include "Matrix.h"
#include "Computation.h"
#include <windows.h>
#include <thread>
#include <vector>

template <typename T>
class ParallelCPU : public Computation<T>
{
public:
    ParallelCPU(
        const Matrix<T> &a, 
        const Matrix<T> &b, 
        const Matrix<T> &c, 
        const Matrix<T> &d, 
        const Matrix<T> &e, 
        Matrix<T> &result);

    std::chrono::duration<double, std::milli> run();

private:
    static DWORD WINAPI computeSection(LPVOID param);

    int numThreads;
    struct ThreadData
    {
        ParallelCPU *instance;
        int start, end;
    };
};
