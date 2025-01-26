#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void matrixCompute(
    __global const real_t* a, 
    __global const real_t* b, 
    __global const real_t* c, 
    __global const real_t* d, 
    __global const real_t* e, 
    __global real_t* result, 
    int width, 
    int height) 
{
    int i = get_global_id(1);
    int j = get_global_id(0);
    for (int k = 0; k < 100000; k++) {
        if (i < height && j < width) {
            int index = i * width + j;
            result[index] = a[index] + b[index] * c[index] - d[index] * e[index];
        }
    }
}
