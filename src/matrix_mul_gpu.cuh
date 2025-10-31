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


// Allocates pinned memory on the host
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

// Allocates device memory synchronously, if a stream is provided, the stream is ignored
struct DeviceMemoryAllocator {
  static constexpr const char* name = "DeviceMemoryAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    void* ptr = nullptr;
    cudaCheck(cudaMalloc(&ptr, size));
    return ptr;
  }

  static inline void deallocate(void* ptr, cudaStream_t stream = nullptr) { cudaCheck(cudaFree(ptr)); }
};

// Allocates device memory asynchronously, requires a stream
struct DeviceAsyncMemoryAllocator {
  static constexpr const char* name = "DeviceAsyncMemoryAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    void* ptr = nullptr;
    cudaCheck(cudaMallocAsync(&ptr, size, stream));
    return ptr;
  }

  static inline void deallocate(void* ptr, cudaStream_t stream) { cudaCheck(cudaFreeAsync(ptr, stream)); }
};

// Does not allocate or release memory, required to pass matrices to kernels
struct NoAllocator {
  static constexpr const char* name = "NoAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    throw std::runtime_error("NoAllocator does not allocate memory");
  }

  static inline void deallocate(void* ptr, cudaStream_t stream = nullptr) {  // no-op, memory was borrowed}
  };
};


// Matrix class, wraps a 2D array of type T, with a memory allocator
template <Numeric T, typename MemoryAllocator>
class Matrix {
 public:
  __host__ Matrix(int width, int height) : width(width), stride(width), height(height), stream(nullptr) {
    data = (T*)MemoryAllocator::allocate(width * height * sizeof(T), nullptr);
  }

  __host__ Matrix(int width, int stride, int height, cudaStream_t stream)
      : width(width), stride(stride), height(height), stream(stream) {
    data = (T*)MemoryAllocator::allocate(width * height * sizeof(T), stream);
  }

  __device__ Matrix(int width, int stride, int height, T* data)
      : width(width), stride(stride), height(height), data(data), stream(nullptr) {}

  __host__ __device__ ~Matrix() { MemoryAllocator::deallocate(data, stream); }

  __host__ __device__ Matrix(const Matrix& other) = delete;
  __host__ __device__ Matrix& operator=(const Matrix&) = delete;

  __host__ __device__ Matrix(Matrix&& other) noexcept = default;
  __host__ __device__ Matrix& operator=(Matrix&& other) noexcept = default;

  Matrix<T, NoAllocator> copy() const { return Matrix<T, NoAllocator>(width, stride, height, data); }

  inline int size() const { return width * stride; }

  // intentionally unsafe for performance reasons
  __device__ __host__ inline T& operator()(int row, int col) { return data[row * stride + col]; }

  // intentionally unsafe for performance reasons
  __device__ __host__ inline const T& operator()(int row, int col) const { return data[row * stride + col]; }

  __device__ __host__ inline Matrix<T, NoAllocator> get_block(int row, int col, int block_size) const {
    if (row < 0 || col < 0 || row >= height || col >= width) {
      return Matrix<T, NoAllocator>(0, 0, 0, (T*)nullptr);
    }
    T* block_data = &data[row * block_size * stride + col * block_size];
    int width = block_size <= (width - col) ? block_size : width - col;
    int height = block_size <= (height - row) ? block_size : height - row;
    Matrix<T, NoAllocator> block(width, stride, height, block_data);
    return block;
  }

  __host__ void randomize(std::default_random_engine& generator) {
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        float val = distribution(generator);
        operator()(row, col) = static_cast<T>(val);
      }
    }
  }

  template <typename OtherAllocator>
  __host__ bool verify(const Matrix<T, OtherAllocator>& other, float tolerance = 0.1) const {
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        float a = static_cast<float>(operator()(row, col));
        float b = static_cast<float>(other.operator()(row, col));
        float diff = std::fabs(a - b);
        if (diff > tolerance) {
          printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at [%d,%d]\n", b, a, diff, row, col);
          return false;
        }
      }
    }
    printf("Matrix verified successfully\n");
    return true;
  }

 public:
  int width;
  int stride;
  int height;
  T* data;
  cudaStream_t stream;
};

#endif  // CUDA_EXERCISES_MATRIX_MUL_GPU_H