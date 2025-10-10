#ifndef CUDA_EXERCISES_UTILS_H
#define CUDA_EXERCISES_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cuda/std/concepts>
#include <memory>

template <typename T>
concept Numeric = cuda::std::integral<T> || cuda::std::floating_point<T> || cuda::std::is_same_v<T, __nv_bfloat16>;

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
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
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
    printf("  Supports cooperative launch: %s (%d)\n", supportsCoopLaunch ? "Yes" : "No", supportsCoopLaunch);
  }

  return deviceProp;
}

class CudaEventRecorder {
 public:
  CudaEventRecorder(const char* operation_name, cudaStream_t stream) : operation_name(operation_name), stream(stream) {
    cudaCheck(cudaEventCreate(&startEvent));
    cudaCheck(cudaEventCreate(&stopEvent));
    cudaCheck(cudaEventRecord(startEvent, stream));
  }

  ~CudaEventRecorder() {
    cudaCheck(cudaEventRecord(stopEvent, stream));
    cudaCheck(cudaEventSynchronize(stopEvent));
    float gpuExecutionTime = 0;
    cudaCheck(cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent));
    printf("GPU time taken to perform %s: %f ms\n", operation_name, gpuExecutionTime);
    cudaCheck(cudaEventDestroy(startEvent));
    cudaCheck(cudaEventDestroy(stopEvent));
  }

  CudaEventRecorder(const CudaEventRecorder&) = delete;
  CudaEventRecorder& operator=(const CudaEventRecorder&) = delete;

  CudaEventRecorder(CudaEventRecorder&& other) noexcept {
    operation_name = other.operation_name;
    stream = other.stream;
    startEvent = other.startEvent;
    stopEvent = other.stopEvent;
    other.operation_name = nullptr;
    other.stream = nullptr;
    other.startEvent = nullptr;
    other.stopEvent = nullptr;
  }

  CudaEventRecorder& operator=(CudaEventRecorder&& other) noexcept {
    if (this != &other) {
      operation_name = other.operation_name;
      stream = other.stream;
      startEvent = other.startEvent;
      stopEvent = other.stopEvent;
      other.operation_name = nullptr;
      other.stream = nullptr;
      other.startEvent = nullptr;
      other.stopEvent = nullptr;
    }
    return *this;
  }

 private:
  const char* operation_name;
  cudaStream_t stream;
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
};

class CudaStream {
 public:
  CudaStream() { cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }

  ~CudaStream() { cudaCheck(cudaStreamDestroy(stream)); }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept {
    stream = other.stream;
    other.stream = nullptr;
  }

  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      cudaCheck(cudaStreamDestroy(stream));
      stream = other.stream;
      other.stream = nullptr;
    }
    return *this;
  }

  CudaEventRecorder record(const char* operation_name) { return CudaEventRecorder(operation_name, stream); }

 public:
  cudaStream_t stream;
};

struct CudaDeleter {
  void operator()(void* ptr) const {
    if (ptr) {
      cudaCheck(cudaFree(ptr));
    }
  }
};

template <Numeric T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

template <Numeric T>
CudaUniquePtr<T> make_cuda_unique(size_t count = 1) {
  T* ptr = nullptr;
  if (count > 0) {
    cudaCheck(cudaMalloc(&ptr, count * sizeof(T)));
  }
  return CudaUniquePtr<T>(ptr);
}

#endif  // CUDA_EXERCISES_UTILS_H