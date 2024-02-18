#ifndef ERROR_H
#define ERROR_H

#include <cuda_runtime.h>


void check(cudaError_t err, const char* const func, const char* const file,
           const int line);

void CHECK_CUDA_ERROR(cudaError_t val);

#endif