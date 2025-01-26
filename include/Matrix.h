#pragma once
#include <vector>
#include <random>
#include <iostream>

template <typename T>
class Matrix
{
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
        "Matrix can only be instantiated with float or double types.");

public:
    int width;
    int height;
    std::vector<T> data;

    Matrix(int w, int h);

    void generate();
    void print(const std::string& name);
};
