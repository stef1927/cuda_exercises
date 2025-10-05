#ifndef CUDA_EXERCISES_MATRIX_H
#define CUDA_EXERCISES_MATRIX_H

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"

// Forward declaration of DeviceMatrix
template <typename T> class DeviceMatrix;

template <typename T> class HostMatrix {
public:
  explicit HostMatrix(int width, int height) : width(width), height(height) {
    allocate();
  }

  ~HostMatrix() { deallocate(); }

  void close() { deallocate(); }

  HostMatrix(const HostMatrix &other)
      : width(other.width), height(other.height) {
    allocate();
    cudaCheck(cudaMemcpy(data, other.data, get_size() * sizeof(T),
                         cudaMemcpyHostToHost));
  }

  HostMatrix &operator=(const HostMatrix &other) {
    if (this != &other) {
      maybe_reallocate(other.width, other.height);
      cudaCheck(cudaMemcpy(data, other.data, get_size() * sizeof(T),
                           cudaMemcpyHostToHost));
    }
    return *this;
  }

  HostMatrix(HostMatrix &&other)
      : width(other.width), height(other.height), data(other.data) {
    other.data = nullptr;
    other.width = 0;
    other.height = 0;
  }

  HostMatrix &operator=(HostMatrix &&other) {
    if (this != &other) {
      deallocate();
      width = other.width;
      height = other.height;
      data = other.data;
      other.data = nullptr;
      other.width = 0;
      other.height = 0;
    }
    return *this;
  }

  DeviceMatrix<T> to_device() const {
    if (data == nullptr) {
      throw std::runtime_error("Cannot copy from a moved-from HostMatrix");
    }
    T *device_data = nullptr;
    cudaCheck(cudaMalloc((void **)&device_data, get_size() * sizeof(T)));
    cudaCheck(cudaMemcpy(device_data, data, get_size() * sizeof(T),
                         cudaMemcpyHostToDevice));
    return DeviceMatrix<T>(width, width, height, device_data);
  }

  void from_device(const DeviceMatrix<T> &other) {
    if (other.data == nullptr) {
      throw std::runtime_error("Cannot copy from a moved-from DeviceMatrix");
    }
    if (data == nullptr) {
      throw std::runtime_error("Cannot copy to a moved-from HostMatrix");
    }
    if (width != other.width || height != other.height) {
      throw std::runtime_error("Cannot copy matrices of different sizes");
    }
    cudaCheck(cudaMemcpy(data, other.data, get_size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
  }

  int get_width() const { return width; }
  int get_height() const { return height; }
  T *get_data() { return data; }
  const T *get_data() const { return data; }
  int get_size() const { return width * height; }

  void randomize(std::default_random_engine &generator) {
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < get_size(); i++) {
      data[i] = distribution(generator);
    }
  }

  bool verify(const HostMatrix<T> &other, float tolerance = 0.1) const {
    for (int i = 0; i < get_size(); i++) {
      float a = data[i];
      float b = other.data[i];
      float diff = std::fabs(a - b);
      if (diff > tolerance) {
        printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at [%d,%d]\n",
               b, a, diff, i / width, i % width);
        return false;
      }
    }
    printf("Matrix verified successfully\n");
    return true;
  }

private:
  void allocate() {
    cudaCheck(cudaMallocHost((void **)&data, get_size() * sizeof(T)));
    memset(data, 0, get_size() * sizeof(T));
  }

  void deallocate() {
    if (data != nullptr) {
      cudaCheck(cudaFreeHost(data));
      data = nullptr;
    }
  }

  void maybe_reallocate(int other_width, int other_height) {
    if (width != other_width || height != other_height) {
      deallocate();
      width = other_width;
      height = other_height;
      allocate();
    }
  }

private:
  int width;
  int height;
  T *data;
};

template <typename T> class DeviceMatrix {
public:
  __host__ __device__ DeviceMatrix(int width, int stride, int height, T *data)
      : width(width), stride(stride), height(height), data(data) {}

  __host__ __device__ ~DeviceMatrix() = default;

  // Copy operations are forbidden
  __host__ __device__ DeviceMatrix(const DeviceMatrix &) = default;
  __host__ __device__ DeviceMatrix &operator=(const DeviceMatrix &) = default;

  __host__ __device__ DeviceMatrix(DeviceMatrix &&other) noexcept = default;
  __host__ __device__ DeviceMatrix &
  operator=(DeviceMatrix &&other) noexcept = default;

  __host__ void close() {
    if (data != nullptr) {
      cudaCheck(cudaFree(data));
      data = nullptr;
    }
  }

  __device__ inline void set_value(T val, int row, int col) {
    if (row >= 0 && row < height && col >= 0 && col < width) {
      data[row * stride + col] = val;
    }
  }

  __device__ inline const T &get_value(int row, int col) const {
    if (row >= 0 && row < height && col >= 0 && col < width) {
      return data[row * stride + col];
    }
    return 0;
  }

  __device__ inline DeviceMatrix<T> get_block(int row, int col,
                                              int block_size) const {
    if (row < 0 || col < 0 || row >= height || col >= width) {
      return DeviceMatrix<T>(0, 0, 0, nullptr);
    }
    T *block_data = &data[row * block_size * stride + col * block_size];
    int width = block_size <= width - col ? block_size : width - col;
    int height = block_size <= height - row ? block_size : height - row;
    DeviceMatrix<T> block(width, stride, height, block_data);
    return block;
  }

public:
  int width;
  int stride;
  int height;
  T *data;
};

#endif // CUDA_EXERCISES_MATRIX_H