#ifndef CUDA_EXERCISES_MATRIX_MUL_CPU_H
#define CUDA_EXERCISES_MATRIX_MUL_CPU_H

#include <cmath>
#include <cstring>
#include <random>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// Layout policy tags (inspired by CUTLASS)
struct RowMajor {
  static constexpr const char* name = "RowMajor";

  static inline int offset(int row, int col, int width, int height) { return row * width + col; }

  static inline int stride(int width, int height) {
    return width;  // Leading dimension
  }
};

struct ColumnMajor {
  static constexpr const char* name = "ColumnMajor";

  static inline int offset(int row, int col, int width, int height) { return col * height + row; }

  static inline int stride(int width, int height) {
    return height;  // Leading dimension
  }
};

template <Numeric T, typename Layout = RowMajor>
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

  Matrix(Matrix&& other) noexcept : width(other.width), height(other.height), data(std::move(other.data)) {
    other.width = 0;
    other.height = 0;
    other.data.clear();
  }

  Matrix& operator=(Matrix&& other) noexcept {
    if (this != &other) {
      width = other.width;
      height = other.height;
      data = std::move(other.data);
      other.width = 0;
      other.height = 0;
      other.data.clear();
    }
    return *this;
  }


  inline int get_width() const { return width; }
  inline int get_height() const { return height; }

  inline int size() const { return width * height; }

  void randomize(std::default_random_engine& generator) {
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int i = 0; i < size(); i++) {
      float val = distribution(generator);
      data[i] = static_cast<T>(val);
    }
  }

  bool verify(const Matrix<T, Layout>& other, float tolerance = 0.1) const {
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        float a = static_cast<float>((*this)(row, col));
        float b = static_cast<float>(other(row, col));
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

  void reset() { std::fill(data.begin(), data.end(), 0); }

  // intentionally unsafe for performance reasons
  inline T operator()(int row, int col) const { return data[Layout::offset(row, col, width, height)]; }

  // intentionally unsafe for performance reasons
  inline T& operator()(int row, int col) { return data[Layout::offset(row, col, width, height)]; }

  static constexpr const char* layout_name() { return Layout::name; }

 private:
  int width;
  int height;
  std::vector<T> data;
};


template <typename T, typename SourceLayout, typename TargetLayout>
Matrix<T, TargetLayout> convert_layout(const Matrix<T, SourceLayout>& source) {
  Matrix<T, TargetLayout> result(source.get_width(), source.get_height());

  for (int row = 0; row < source.get_height(); row++) {
    for (int col = 0; col < source.get_width(); col++) {
      result(row, col) = source(row, col);
    }
  }

  return result;
}

template <typename T>
Matrix<T, ColumnMajor> to_column_major(const Matrix<T, RowMajor>& source) {
  return convert_layout<T, RowMajor, ColumnMajor>(source);
}

template <typename T>
Matrix<T, RowMajor> to_row_major(const Matrix<T, ColumnMajor>& source) {
  return convert_layout<T, ColumnMajor, RowMajor>(source);
}

#endif  // CUDA_EXERCISES_MATRIX_MUL_CPU_H