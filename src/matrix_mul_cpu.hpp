#ifndef CUDA_EXERCISES_MATRIX_MUL_CPU_H
#define CUDA_EXERCISES_MATRIX_MUL_CPU_H

#include <cmath>
#include <cstring>
#include <random>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;


template <Numeric T>
class Matrix {
 public:
  explicit Matrix(int width, int height) : width(width), height(height) { data.resize(width * height); }

  Matrix(const Matrix& other) : width(other.width), height(other.height) {
    data.resize(width * height);
    std::copy(other.data.begin(), other.data.end(), data.begin());
  }

  Matrix& operator=(const Matrix& other) {
    if (this != &other) {
      width = other.width;
      height = other.height;
      data.resize(width * height);
      std::copy(other.data.begin(), other.data.end(), data.begin());
    }
    return *this;
  }

  Matrix(Matrix&& other) : width(other.width), height(other.height), data(other.data) {
    other.data = nullptr;
    other.width = 0;
    other.height = 0;
  }

  Matrix& operator=(Matrix&& other) {
    if (this != &other) {
      width = other.width;
      height = other.height;
      data = other.data;
      other.data = nullptr;
      other.width = 0;
      other.height = 0;
    }
    return *this;
  }


  int get_width() const { return width; }
  int get_height() const { return height; }
  T* get_data() { return data; }
  const T* get_data() const { return data; }
  int get_size() const { return width * height; }

  void randomize(std::default_random_engine& generator) {
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int i = 0; i < get_size(); i++) {
      float val = distribution(generator);
      data[i] = static_cast<T>(val);
    }
  }

  bool verify(const Matrix<T>& other, float tolerance = 0.1) const {
    for (int i = 0; i < get_size(); i++) {
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

  void print_row(const std::string& name, int row) const {
    printf("%s Row %d: [", name.c_str(), row);
    for (int i = 0; i < width; i++) {
      printf("%5.2f ", static_cast<float>(data[row * width + i]));
    }
    printf("]\n");
  }

  inline T get_value(int row, int col) const {
    if (row >= 0 && row < height && col >= 0 && col < width) {
      return data[row * width + col];
    }
    return 0;
  }

  inline void set_value(T val, int row, int col) {
    if (row >= 0 && row < height && col >= 0 && col < width) {
      data[row * width + col] = val;
    }
  }

 private:
  int width;
  int height;
  std::vector<T> data;
};

#endif  // CUDA_EXERCISES_MATRIX_MUL_CPU_H