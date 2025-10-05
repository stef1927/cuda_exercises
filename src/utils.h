#ifndef CUDA_EXERCISES_UTILS_H
#define CUDA_EXERCISES_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>

inline void __cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
      printf("[CUDA ERROR] at file %s:%d: %s - %s\n", file, line,
            cudaGetErrorName(error), cudaGetErrorString(error));
      exit(1);
    }
  }

#define cudaCheck(err) (__cudaCheck(err, __FILE__, __LINE__))

#endif // CUDA_EXERCISES_UTILS_H