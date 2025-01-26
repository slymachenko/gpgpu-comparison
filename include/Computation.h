#pragma once
#include "Matrix.h"
#include <chrono>

template <typename T>
class Computation
{
public:
    Computation(
        const Matrix<T> &a,
        const Matrix<T> &b,
        const Matrix<T> &c,
        const Matrix<T> &d,
        const Matrix<T> &e,
        Matrix<T> &result)
        : a(a), b(b), c(c), d(d), e(e), result(result), width(a.width), height(a.height) {}

    virtual std::chrono::duration<double, std::milli> run() = 0;
    virtual ~Computation() = default;

protected:
    const Matrix<T> &a, &b, &c, &d, &e;
    Matrix<T> &result;
    int width;
    int height;
};
