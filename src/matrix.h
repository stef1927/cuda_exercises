#ifndef CUDA_EXERCISES_MATRIX_H
#define CUDA_EXERCISES_MATRIX_H

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"

enum class MatrixType { HOST, DEVICE };

template <typename T> class Matrix {
public:
  explicit Matrix(int width, int height, MatrixType type)
      : width(width), height(height), type(type) {
    allocate(true);
  }

  ~Matrix() { deallocate(); }

  Matrix(const Matrix &other)
      : width(other.width), height(other.height), type(other.type) {
    allocate(true);
    copy(other);
  }

  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      copy(other);
    }
    return *this;
  }

  Matrix(Matrix &&other)
      : width(other.width), height(other.height), type(other.type),
        data(other.data) {
    other.data = nullptr;
    other.width = 0;
    other.height = 0;
  }

  Matrix &operator=(Matrix &&other) {
    if (this != &other) {
      if (type != other.type) {
        throw std::invalid_argument("Invalid matrix type");
      }
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

  void set_value(T val, int row, int col) { data[row * width + col] = val; }

  const T &get_value(int row, int col) const { return data[row * width + col]; }

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

  bool verify(const Matrix &other, float tolerance = 0.1) const {
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
  void allocate(bool zero = false) {
    if (type == MatrixType::HOST) {
      cudaCheck(cudaMallocHost((void **)&data, get_size() * sizeof(T)));
      if (zero) {
        memset(data, 0, get_size() * sizeof(T));
      }
    } else if (type == MatrixType::DEVICE) {
      cudaCheck(cudaMalloc((void **)&data, get_size() * sizeof(T)));
      if (zero) {
        cudaCheck(cudaMemset(data, 0, get_size() * sizeof(T)));
      }
    } else {
      throw std::invalid_argument("Invalid matrix type");
    }
  }

  void deallocate() {
    if (type == MatrixType::HOST) {
      cudaCheck(cudaFreeHost(data));
    } else if (type == MatrixType::DEVICE) {
      cudaCheck(cudaFree(data));
    }
  }

  void maybe_reallocate(const Matrix &other) {
    if (width != other.width || height != other.height) {
      deallocate();
      width = other.width;
      height = other.height;
      allocate(false);
    }
  }

  void copy(const Matrix &other) {
    maybe_reallocate(other);
    if (type == MatrixType::HOST) {
      if (other.type == MatrixType::HOST) {
        memcpy(data, other.data, get_size() * sizeof(T));
      } else if (other.type == MatrixType::DEVICE) {
        cudaCheck(cudaMemcpy(data, other.data, get_size() * sizeof(T),
                             cudaMemcpyDeviceToHost));
      } else {
        throw std::invalid_argument("Invalid matrix type");
      }
    } else if (type == MatrixType::DEVICE) {
      if (other.type == MatrixType::HOST) {
        cudaCheck(cudaMemcpy(data, other.data, get_size() * sizeof(T),
                             cudaMemcpyDeviceToHost));
      } else if (other.type == MatrixType::DEVICE) {
        cudaCheck(cudaMemcpy(data, other.data, get_size() * sizeof(T),
                             cudaMemcpyDeviceToDevice));
      } else {
        throw std::invalid_argument("Invalid matrix type");
      }
    } else {
      throw std::invalid_argument("Invalid matrix type");
    }
  }

private:
  int width;
  int height;
  MatrixType type;
  T *data;
};

#endif // CUDA_EXERCISES_MATRIX_H