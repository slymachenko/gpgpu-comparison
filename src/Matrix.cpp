#include "Matrix.h"

template <typename T>
Matrix<T>::Matrix(int w, int h) : width(w), height(h), data(w * h)
{
    generate();
}

template <typename T>
void Matrix<T>::generate()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(static_cast<T>(1.0), static_cast<T>(10.0));
    for (int i = 0; i < width * height; i++)
    {
        data[i] = dis(gen);
    }
}

template <typename T>
void Matrix<T>::print(const std::string & name)
{
    std::cout << "Matrix " << name << " (" << width << "x" << height << "):" << std::endl;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << data[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template class Matrix<float>;
template class Matrix<double>;
