#ifndef CUDA_EXERCISES_UTILS_H
#define CUDA_EXERCISES_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

inline void __cudaCheck(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d: %s - %s\n", file, line, cudaGetErrorName(error), cudaGetErrorString(error));
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
    printf("  CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total amount of shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Supports Zero Copy mapped memory: %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Supports overlapping GPU compute and memory copies: %s (%d)\n",
           deviceProp.asyncEngineCount > 0 ? "Yes" : "No", deviceProp.asyncEngineCount);
    printf("  Supports managed memory (UMA): %s (%d)\n", deviceProp.managedMemory > 0 ? "Yes" : "No",
           deviceProp.managedMemory);
    printf("  Supports UMA concurrent access: %s (%d)\n", deviceProp.concurrentManagedAccess > 0 ? "Yes" : "No",
           deviceProp.concurrentManagedAccess);
    printf("  Supports UMA for system memory: %s (%d)\n", deviceProp.pageableMemoryAccess > 0 ? "Yes" : "No",
           deviceProp.pageableMemoryAccess);
  }

  return deviceProp;
}
#endif  // CUDA_EXERCISES_UTILS_H