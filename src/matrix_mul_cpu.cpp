#include "matrix_mul_cpu.hpp"

#include <format>
#include <random>

#include "argparse.hpp"
#include "cpp_utils.hpp"

static std::default_random_engine generator(786);

struct Args {
  int width;
  int height;
  int block_size;
  int num_runs;
};

template <Numeric T, typename LayoutA, typename LayoutB, typename LayoutC>
void matrixMulSerial(const Matrix<T, LayoutA>& A, const Matrix<T, LayoutB>& B, Matrix<T, LayoutC>& C) {
  Timer timer("serial matrix multiplication " + std::string(LayoutA::name) + " x " + std::string(LayoutB::name) +
              " -> " + std::string(LayoutC::name));
  for (int row = 0; row < A.get_height(); row++) {
    for (int col = 0; col < B.get_width(); col++) {
      T sum = 0.0;
      for (int k = 0; k < A.get_width(); k++) {
        sum += A(row, k) * B(k, col);
      }
      C(row, col) = sum;
    }
  }
}

template <Numeric T, typename LayoutA, typename LayoutB, typename LayoutC>
void matrixMulParallel(const Matrix<T, LayoutA>& A, const Matrix<T, LayoutB>& B, Matrix<T, LayoutC>& C, int num_runs) {
  std::string name =
      std::format("parallel matrix multiplication {} x {} -> {}", LayoutA::name, LayoutB::name, LayoutC::name);
  Timer timer(name, num_runs);
  for (int i = 0; i < num_runs; i++) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < A.get_height(); row++) {
      for (int col = 0; col < B.get_width(); col++) {
        T sum = 0.0;
#pragma omp simd reduction(+ : sum)
        for (int k = 0; k < A.get_width(); k++) {
          sum += A(row, k) * B(k, col);
        }
        C(row, col) = sum;
      }
    }
  }
  timer.stop();
}


template <Numeric T, typename LayoutA, typename LayoutB, typename LayoutC>
void matrixMulParallelTiled(const Matrix<T, LayoutA>& A, const Matrix<T, LayoutB>& B, Matrix<T, LayoutC>& C,
                            int block_size, int num_runs) {
  std::string name =
      std::format("parallel matrix multiplication tiled {} x {} -> {}", LayoutA::name, LayoutB::name, LayoutC::name);
  Timer timer(name, num_runs, false);
  for (int i = 0; i < num_runs; i++) {
    C.reset();
    timer.start();
#pragma omp parallel for collapse(2) schedule(static)
    for (int row_block = 0; row_block < A.get_height(); row_block += block_size) {
      for (int col_block = 0; col_block < B.get_width(); col_block += block_size) {
        for (int k_block = 0; k_block < A.get_width(); k_block += block_size) {
          int end_row = std::min(row_block + block_size, A.get_height());
          int end_col = std::min(col_block + block_size, B.get_width());
          int min_k = std::min(k_block + block_size, A.get_width());
          for (int row = row_block; row < end_row; row++) {
            for (int col = col_block; col < end_col; col++) {
              T sum = 0.0;
#pragma omp simd reduction(+ : sum)
              for (int k = k_block; k < min_k; k++) {
                sum += A(row, k) * B(k, col);
              }
              C(row, col) += sum;
            }
          }
        }
      }
    }
    timer.stop();
  }
}

int parse_args(int argc, char* argv[], Args& args) {
  argparse::ArgumentParser program("matrix_mul");
  std::string kernel_type;

  program.add_argument("--width")
      .help("width of the matrix")
      .scan<'i', int>()
      .default_value(1024)
      .store_into(args.width);

  program.add_argument("--height")
      .help("height of the matrix")
      .scan<'i', int>()
      .default_value(1024)
      .store_into(args.height);

  program.add_argument("--block-size")
      .help("block size")
      .scan<'i', int>()
      .default_value(32)
      .store_into(args.block_size);

  program.add_argument("--num-runs")
      .help("number of runs")
      .scan<'i', int>()
      .default_value(100)
      .store_into(args.num_runs);

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
  printf("Number of runs: %d\n", args.num_runs);

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

  auto B_col_major = to_column_major(B);

  // matrixMulSerial(A, B, C_ref);

  matrixMulSerial(A, B_col_major, C_ref);
  // C.verify(C_ref);
  // C.reset();

  // matrixMulParallel(A, B, C, args.num_runs);
  // C.verify(C_ref);
  // C.reset();

  matrixMulParallel(A, B_col_major, C, args.num_runs);
  C.verify(C_ref);
  C.reset();

  matrixMulParallelTiled(A, B_col_major, C, args.block_size, args.num_runs);
  C.verify(C_ref);
  C.reset();

  return 0;
}