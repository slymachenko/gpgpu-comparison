#include "ParallelCPU.h"

template <typename T>
ParallelCPU<T>::ParallelCPU(const Matrix<T> &a, const Matrix<T> &b, const Matrix<T> &c, const Matrix<T> &d, const Matrix<T> &e, Matrix<T> &result)
    : Computation<T>(a, b, c, d, e, result)
{
    numThreads = std::thread::hardware_concurrency();
}

template <typename T>
DWORD WINAPI ParallelCPU<T>::computeSection(LPVOID param)
{
    ThreadData *data = static_cast<ThreadData *>(param);
    ParallelCPU *instance = data->instance;
    for (int k = 0; k < 100000; k++)
    {
        for (int i = data->start; i < data->end; i++)
        {
            for (int j = 0; j < instance->width; j++)
            {
                int index = i * instance->width + j;
                instance->result.data[index] = instance->a.data[index] + instance->b.data[index] * instance->c.data[index] - instance->d.data[index] * instance->e.data[index];
            }
        }
    }
    return 0;
}

template <typename T>
std::chrono::duration<double, std::milli> ParallelCPU<T>::run()
{
    HANDLE *threads = new HANDLE[numThreads];
    ThreadData *threadData = new ThreadData[numThreads];
    int chunkSize = height / numThreads;

    for (int i = 0; i < numThreads; i++)
    {
        threadData[i].instance = this;
        threadData[i].start = i * chunkSize;
        threadData[i].end = (i == numThreads - 1) ? height : threadData[i].start + chunkSize;
        threads[i] = CreateThread(NULL, 0, computeSection, &threadData[i], 0, NULL);
    }

    auto computeStart = std::chrono::high_resolution_clock::now();
    WaitForMultipleObjects(numThreads, threads, TRUE, INFINITE);
    auto computeEnd = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++)
    {
        CloseHandle(threads[i]);
    }

    delete[] threads;
    delete[] threadData;

    return computeEnd - computeStart;
}

template class ParallelCPU<float>;
template class ParallelCPU<double>;
