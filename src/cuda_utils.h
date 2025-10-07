#ifndef CUDA_EXERCISES_UTILS_H
#define CUDA_EXERCISES_UTILS_H

#include <cstdio>
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

inline cudaDeviceProp getDeviceProperties(int dev = 0, bool print = true) {
  cudaDeviceProp deviceProp;
  cudaCheck(cudaGetDeviceProperties(&deviceProp, dev));
  if (print) {
    printf("Device %d: %s\n", dev, deviceProp.name);
    printf("  Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    printf("  CUDA Capability Major/Minor version number: %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("  Total amount of shared memory per block: %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Maximum number of threads per block: %d\n",
           deviceProp.maxThreadsPerBlock);
  }

  return deviceProp;
}
#endif // CUDA_EXERCISES_UTILS_H