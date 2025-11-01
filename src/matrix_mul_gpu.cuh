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
struct DeviceSyncMemoryAllocator {
  static constexpr const char* name = "DeviceSyncMemoryAllocator";

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

  static inline void* allocate(size_t size, cudaStream_t stream) {
    void* ptr = nullptr;
    cudaCheck(cudaMallocAsync(&ptr, size, stream));
    return ptr;
  }

  static inline void deallocate(void* ptr, cudaStream_t stream) { cudaCheck(cudaFreeAsync(ptr, stream)); }
};

// Does not allocate or release memory, this is used to pass matrices to CUDA kernels to to create block sub-matrices in
// kernels
struct NoAllocator {
  static constexpr const char* name = "NoAllocator";

  static inline void* allocate(size_t size, cudaStream_t stream = nullptr) {
    throw std::runtime_error("NoAllocator does not allocate memory");
  }

  static inline void deallocate(void* ptr, cudaStream_t stream = nullptr) {  // no-op, memory was borrowed}
  };
};


// Row major format for matrices, the default layout, organizes memory in a row-major order
struct RowMajor {
  static constexpr const char* name = "RowMajor";

  __host__ __device__ explicit RowMajor(int width, int height) : width(width), height(height), stride(width) {}

  __host__ __device__ explicit RowMajor(int width, int height, int stride)
      : width(width), height(height), stride(stride) {}

  __host__ __device__ inline int offset(int row, int col, int block_size = 1) const {
    return row * stride * block_size + col * block_size;
  }

  __host__ __device__ inline int size() const { return height * stride; }

  int width;
  int height;
  int stride;
};

// Column major format for matrices, organizes memory in a column-major order
struct ColumnMajor {
  static constexpr const char* name = "ColumnMajor";

  __host__ __device__ explicit ColumnMajor(int width, int height) : width(width), height(height), stride(height) {}

  __host__ __device__ explicit ColumnMajor(int width, int height, int stride)
      : width(width), height(height), stride(stride) {}

  __host__ __device__ inline int offset(int row, int col, int block_size = 1) const {
    return col * stride * block_size + row * block_size;
  }

  __host__ __device__ inline int size() const { return width * stride; }

  int width;
  int height;
  int stride;
};


// Matrix class, wraps a 2D array of type T, with a memory allocator and a layout policy
// Can be used on the host and on the device, some functions are limited to the host especially
// the constructors: memory is allocated when the matrix is created on the host but not when
// sub-matrices are created in kernels
template <Numeric T, typename MemoryAllocator, typename Layout = RowMajor>
class Matrix {
 public:
  __host__ Matrix(int width, int height) : layout(Layout(width, height)), stream(nullptr) {
    data = (T*)MemoryAllocator::allocate(layout.size() * sizeof(T), stream);
  }

  __host__ Matrix(int width, int stride, int height, cudaStream_t stream)
      : layout(Layout(width, height, stride)), stream(stream) {
    data = (T*)MemoryAllocator::allocate(layout.size() * sizeof(T), stream);
  }

  __device__ Matrix(int width, int stride, int height, T* data)
      : layout(Layout(width, height, stride)), data(data), stream(nullptr) {}

  __host__ __device__ ~Matrix() { MemoryAllocator::deallocate(data, stream); }

  __host__ __device__ Matrix(const Matrix& other) = delete;
  __host__ __device__ Matrix& operator=(const Matrix&) = delete;

  __host__ __device__ Matrix(Matrix&& other) noexcept = default;
  __host__ __device__ Matrix& operator=(Matrix&& other) noexcept = default;

  Matrix<T, NoAllocator, Layout> copy() const {
    return Matrix<T, NoAllocator, Layout>(layout.width, layout.stride, layout.height, data);
  }

  __host__ __device__ inline int size() const { return layout.size(); }
  __host__ __device__ inline int width() const { return layout.width; }
  __host__ __device__ inline int height() const { return layout.height; }
  __host__ __device__ inline int stride() const { return layout.stride; }

  // intentionally unsafe for performance reasons
  __host__ __device__ inline T& operator()(int row, int col) { return data[layout.offset(row, col)]; }

  // intentionally unsafe for performance reasons
  __host__ __device__ inline const T& operator()(int row, int col) const { return data[layout.offset(row, col)]; }

  __host__ __device__ inline Matrix<T, NoAllocator, Layout> get_block(int row, int col, int block_size) const {
    T* block_data = &data[layout.offset(row, col, block_size)];
    int width = block_size <= (layout.width - col) ? block_size : layout.width - col;
    int height = block_size <= (layout.height - row) ? block_size : layout.height - row;
    Matrix<T, NoAllocator, Layout> block(width, layout.stride, height, block_data);
    return block;
  }

  __host__ void randomize(std::default_random_engine& generator) {
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int row = 0; row < layout.height; row++) {
      for (int col = 0; col < layout.width; col++) {
        float val = distribution(generator);
        operator()(row, col) = static_cast<T>(val);
      }
    }
  }

  template <typename OtherAllocator, typename OtherLayout>
  __host__ bool verify(const Matrix<T, OtherAllocator, OtherLayout>& other, float tolerance = 0.1) const {
    for (int row = 0; row < layout.height; row++) {
      for (int col = 0; col < layout.width; col++) {
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
  T* data;
  Layout layout;
  cudaStream_t stream;
};


// Convert a matrix from one layout to another
template <typename T, typename MemoryAllocator, typename SourceLayout, typename TargetLayout>
__host__ Matrix<T, MemoryAllocator, TargetLayout> convert_layout(
    const Matrix<T, MemoryAllocator, SourceLayout>& source) {
  Matrix<T, MemoryAllocator, TargetLayout> result(source.width(), source.stride(), source.height(), source.stream);

  for (int row = 0; row < source.height(); row++) {
    for (int col = 0; col < source.width(); col++) {
      result(row, col) = source(row, col);
    }
  }

  return result;
}


// Convert a matrix from row major to column major
template <Numeric T, typename MemoryAllocator>
__host__ Matrix<T, MemoryAllocator, ColumnMajor> to_column_major(const Matrix<T, MemoryAllocator, RowMajor>& source) {
  return convert_layout<T, MemoryAllocator, RowMajor, ColumnMajor>(source);
}


// Convert a matrix from column major to row major
template <Numeric T, typename MemoryAllocator>
__host__ Matrix<T, MemoryAllocator, RowMajor> to_row_major(const Matrix<T, MemoryAllocator, ColumnMajor>& source) {
  return convert_layout<T, MemoryAllocator, ColumnMajor, RowMajor>(source);
}

#endif  // CUDA_EXERCISES_MATRIX_MUL_GPU_H