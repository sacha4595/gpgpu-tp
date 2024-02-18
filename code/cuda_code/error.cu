#include <cuda_runtime.h>
#include <stdio.h>

void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        printf("CUDA Runtime Error at: %s:%d\n", file, line);
        // 
        printf("%s %s\n", cudaGetErrorString(err), func);
    }
}

void CHECK_CUDA_ERROR(cudaError_t val)
{
    check(val, "error", __FILE__, __LINE__);
}