#include "matrix_mul_cpu.hpp"

#include <random>

#include "argparse.hpp"
#include "cpp_utils.hpp"

static std::default_random_engine generator(786);

struct Args {
  int width;
  int height;
  int block_size;
};

template <Numeric T>
void matrixMulSerial(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
  Timer timer("serial matrix multiplication");
  NVTXScopedRange fn("matrixMulSerial");
  for (int row = 0; row < A.get_height(); row++) {
    for (int col = 0; col < B.get_width(); col++) {
      T sum = 0.0;
      for (int k = 0; k < A.get_width(); k++) {
        sum += A.get_value(row, k) * B.get_value(k, col);
      }
      C.set_value(sum, row, col);
    }
  }
}

template <Numeric T>
void matrixMulParallel(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int block_size) {
  Timer timer("parallel matrix multiplication");
  NVTXScopedRange fn("matrixMulParallel");
#pragma omp parallel for collapse(2) schedule(static)
  for (int row = 0; row < A.get_height(); row++) {
    for (int col = 0; col < B.get_width(); col++) {
      T sum = 0.0;
      for (int k = 0; k < A.get_width(); k++) {
        sum += A.get_value(row, k) * B.get_value(k, col);
      }
      C.set_value(sum, row, col);
    }
  }
}

int parse_args(int argc, char* argv[], Args& args) {
  argparse::ArgumentParser program("matrix_mul");
  std::string kernel_type;

  program.add_argument("--width")
      .help("width of the matrix")
      .scan<'i', int>()
      .default_value(256)
      .store_into(args.width);

  program.add_argument("--height")
      .help("height of the matrix")
      .scan<'i', int>()
      .default_value(256)
      .store_into(args.height);

  program.add_argument("--block-size")
      .help("block size")
      .scan<'i', int>()
      .default_value(32)
      .store_into(args.block_size);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }


  // current limitations
  if (args.width != args.height) {
    std::cerr << "Width and height must be equal" << std::endl;
    return 1;
  }

  if (args.width % args.block_size != 0) {
    std::cerr << "Width must be divisible by block size" << std::endl;
    return 1;
  }

  if (args.height % args.block_size != 0) {
    std::cerr << "Height must be divisible by block size" << std::endl;
    return 1;
  }

  printf("Arguments:\n");
  printf("Width: %d\n", args.width);
  printf("Height: %d\n", args.height);
  printf("Block size: %d\n", args.block_size);

  return 0;
}

int main(int argc, char* argv[]) {
  Args args;
  if (parse_args(argc, argv, args) != 0) {
    return 1;
  }

  Matrix<float> A(args.width, args.height);
  Matrix<float> B(args.width, args.height);
  Matrix<float> C(args.width, args.height);
  Matrix<float> C_ref(args.width, args.height);

  A.randomize(generator);
  B.randomize(generator);

  printf("Performaing serial matrix multiplication\n");
  matrixMulSerial(A, B, C_ref);

  printf("Performaing parallel matrix multiplication\n");
  matrixMulParallel(A, B, C, args.block_size);

  printf("Verifying results\n");
  C.verify(C_ref);

  return 0;
}