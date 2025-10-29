#ifndef CUDA_EXERCISES_MATRIX_MUL_GPU_H
#define CUDA_EXERCISES_MATRIX_MUL_GPU_H

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include "cuda_utils.cuh"


struct HostMemoryAllocator {
  static constexpr const char* name = "HostMemoryAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    void* ptr = nullptr;
    cudaCheck(cudaMallocHost(&ptr, size));
    memset(ptr, 0, size);
    return ptr;
  }

  static inline void deallocate(void* ptr, cudaStream_t stream = nullptr) { cudaCheck(cudaFreeHost(ptr)); }
};


struct DeviceMemoryAllocator {
  static constexpr const char* name = "DeviceMemoryAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    void* ptr = nullptr;
    cudaCheck(cudaMalloc(&ptr, size));
    return ptr;
  }

  static inline void deallocate(void* ptr, cudaStream_t stream = nullptr) { cudaCheck(cudaFree(ptr)); }
};


struct DeviceAsyncMemoryAllocator {
  static constexpr const char* name = "DeviceAsyncMemoryAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    void* ptr = nullptr;
    cudaCheck(cudaMallocAsync(&ptr, size, stream));
    return ptr;
  }

  static inline void deallocate(void* ptr, cudaStream_t stream) { cudaCheck(cudaFreeAsync(ptr, stream)); }
};

struct NoAllocator {
  static constexpr const char* name = "NoAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    throw std::runtime_error("NoAllocator does not allocate memory");
  }

  static inline void deallocate(void* ptr, cudaStream_t stream = nullptr) {  // no-op, memory was borrowed}
  };
};


template <Numeric T, typename MemoryAllocator = HostMemoryAllocator>
class HostMatrix {
 public:
  explicit HostMatrix(int width, int height) : width(width), height(height) {
    data = (T*)MemoryAllocator::allocate(size() * sizeof(T));
  }

  ~HostMatrix() { MemoryAllocator::deallocate(data); }

  HostMatrix(const HostMatrix& other) = delete;

  HostMatrix& operator=(const HostMatrix& other) = delete;

  HostMatrix(HostMatrix&& other) noexcept = default;

  HostMatrix& operator=(HostMatrix&& other) noexcept = default;

  inline const int size() const { return width * height; }

  void randomize(std::default_random_engine& generator) {
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int i = 0; i < size(); i++) {
      float val = distribution(generator);
      data[i] = static_cast<T>(val);
    }
  }

  bool verify(const HostMatrix<T, MemoryAllocator>& other, float tolerance = 0.1) const {
    for (int i = 0; i < size(); i++) {
      float a = static_cast<float>(data[i]);
      float b = static_cast<float>(other.data[i]);
      float diff = std::fabs(a - b);
      if (diff > tolerance) {
        printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at [%d,%d]\n", b, a, diff, i / width, i % width);
        return false;
      }
    }
    printf("Matrix verified successfully\n");
    return true;
  }

  // intentionally unsafe for performance reasons
  inline T& operator()(int row, int col) { return data[row * width + col]; }

  // intentionally unsafe for performance reasons
  inline const T& operator()(int row, int col) const { return data[row * width + col]; }

 public:
  int width;
  int height;
  T* data;
};

template <Numeric T, typename MemoryAllocator = DeviceAsyncMemoryAllocator>
class DeviceMatrix {
 public:
  __host__ DeviceMatrix(int width, int stride, int height, cudaStream_t stream)
      : width(width), stride(stride), height(height), stream(stream) {
    data = (T*)MemoryAllocator::allocate(width * height * sizeof(T), stream);
  }

  __device__ DeviceMatrix(int width, int stride, int height, T* data)
      : width(width), stride(stride), height(height), data(data), stream(nullptr) {}

  __host__ __device__ ~DeviceMatrix() { MemoryAllocator::deallocate(data, stream); }

  __host__ __device__ DeviceMatrix(const DeviceMatrix& other) = delete;
  __host__ __device__ DeviceMatrix& operator=(const DeviceMatrix&) = delete;

  __host__ __device__ DeviceMatrix(DeviceMatrix&& other) noexcept = default;
  __host__ __device__ DeviceMatrix& operator=(DeviceMatrix&& other) noexcept = default;

  DeviceMatrix<T, NoAllocator> copy() const { return DeviceMatrix<T, NoAllocator>(width, stride, height, data); }

  // intentionally unsafe for performance reasons
  __device__ __host__ inline T& operator()(int row, int col) { return data[row * stride + col]; }

  // intentionally unsafe for performance reasons
  __device__ __host__ inline const T& operator()(int row, int col) const { return data[row * stride + col]; }

  __device__ __host__ inline DeviceMatrix<T, NoAllocator> get_block(int row, int col, int block_size) const {
    if (row < 0 || col < 0 || row >= height || col >= width) {
      return DeviceMatrix<T, NoAllocator>(0, 0, 0, (T*)nullptr);
    }
    T* block_data = &data[row * block_size * stride + col * block_size];
    int width = block_size <= (width - col) ? block_size : width - col;
    int height = block_size <= (height - row) ? block_size : height - row;
    DeviceMatrix<T, NoAllocator> block(width, stride, height, block_data);
    return block;
  }

 public:
  int width;
  int stride;
  int height;
  T* data;
  cudaStream_t stream;
};

#endif  // CUDA_EXERCISES_MATRIX_MUL_GPU_H