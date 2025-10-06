#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "argparse.hpp"
#include "matrix.h"
#include "utils.h"

typedef __nv_bfloat16 bf16;

static std::default_random_engine generator(786);

enum class KernelType { NAIVE, TILED };

struct Args {
  int width;
  int height;
  int block_size;
  int num_runs;
  KernelType kernel_type;
};

template <typename T>
void matrixMulCpu(const HostMatrix<T> &A, const HostMatrix<T> &B,
                  HostMatrix<T> &C) {
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

template <typename T>
__global__ void matrixMulNaiveKernel(DeviceMatrix<T> A, DeviceMatrix<T> B,
                                     DeviceMatrix<T> C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < C.height && col < C.width) {
    T sum = 0.0;
    for (int k = 0; k < A.width; ++k) {
      sum += A.get_value(row, k) * B.get_value(k, col);
    }

    C.set_value(sum, row, col);
  }
}

template <typename T>
__global__ void matrixMulTiledKernel(DeviceMatrix<T> A, DeviceMatrix<T> B,
                                     DeviceMatrix<T> C, int block_size) {

  const int blockRow = blockIdx.y;
  const int row = threadIdx.y;
  const int blockCol = blockIdx.x;
  const int col = threadIdx.x;

  extern __shared__ T smem[];
  T *As = &smem[0];
  T *Bs = &smem[block_size * block_size];

  DeviceMatrix<T> Csub = C.get_block(blockRow, blockCol, block_size);
  T sum = 0.0;
  for (int b = 0; b < (A.width / block_size); ++b) {
    DeviceMatrix<T> Asub = A.get_block(blockRow, b, block_size);
    DeviceMatrix<T> Bsub = B.get_block(b, blockCol, block_size);

    As[row * block_size + col] = Asub.get_value(row, col);
    Bs[row * block_size + col] = Bsub.get_value(row, col);

    __syncthreads();

    for (int k = 0; k < block_size; ++k) {
      sum += As[row * block_size + k] * Bs[k * block_size + col];
    }

    __syncthreads();
  }
  Csub.set_value(sum, row, col);
}

template <typename T>
float run_kernel(KernelType kernelType, DeviceMatrix<T> &dA,
                 DeviceMatrix<T> &dB, DeviceMatrix<T> &dC, cudaStream_t stream,
                 int block_size) {
  cudaEvent_t startEvent, stopEvent;
  cudaCheck(cudaEventCreate(&startEvent));
  cudaCheck(cudaEventCreate(&stopEvent));

  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid((dA.width + dimBlock.x - 1) / dimBlock.x,
               (dA.height + dimBlock.y - 1) / dimBlock.y);

  cudaCheck(cudaEventRecord(startEvent, 0));
  if (kernelType == KernelType::NAIVE) {
    matrixMulNaiveKernel<T><<<dimGrid, dimBlock, 0, stream>>>(dA, dB, dC);
  } else if (kernelType == KernelType::TILED) {
    int shared_mem_size = 2 * block_size * block_size * sizeof(T);
    matrixMulTiledKernel<T><<<dimGrid, dimBlock, shared_mem_size, stream>>>(
        dA, dB, dC, block_size);
  } else {
    throw std::runtime_error("Invalid kernel type");
  }
  cudaCheck(cudaEventRecord(stopEvent, 0));
  cudaCheck(cudaEventSynchronize(stopEvent));

  float gpuExecutionTime = 0;
  cudaCheck(cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent));
  return gpuExecutionTime;
}

int parse_args(int argc, char *argv[], Args &args) {
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

  program.add_argument("--num-runs")
      .help("number of runs")
      .scan<'i', int>()
      .default_value(100)
      .store_into(args.num_runs);

  program.add_argument("--kernel-type")
      .help("kernel type")
      .default_value("naive")
      .store_into(kernel_type);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if (kernel_type == "naive") {
    args.kernel_type = KernelType::NAIVE;
  } else if (kernel_type == "tiled") {
    args.kernel_type = KernelType::TILED;
  } else {
    std::cerr << "Invalid kernel type: " << kernel_type << std::endl;
    return 1;
  }

  // current limitation, for later on, we need different width and height
  // for A and B and if width != height, then A.width must be equal to B.height
  // (k)
  if (args.width != args.height) {
    std::cerr << "Width and height must be equal" << std::endl;
    return 1;
  }

  printf("Arguments:\n");
  printf("Width: %d\n", args.width);
  printf("Height: %d\n", args.height);
  printf("Block size: %d\n", args.block_size);
  printf("Number of runs: %d\n", args.num_runs);
  printf("Kernel type: %s\n",
         args.kernel_type == KernelType::NAIVE ? "naive" : "tiled");

  return 0;
}

int main(int argc, char *argv[]) {

  Args args;
  if (parse_args(argc, argv, args) != 0) {
    return 1;
  }

  HostMatrix<bf16> A(args.width, args.height);
  HostMatrix<bf16> B(args.width, args.height);
  HostMatrix<bf16> C(args.width, args.height);
  HostMatrix<bf16> C_ref(args.width, args.height);

  A.randomize(generator);
  B.randomize(generator);

  printf("Performaing matrix multiplication on CPU\n");
  matrixMulCpu<bf16>(A, B, C_ref);

  printf("Initializing device matrices\n");
  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  DeviceMatrix<bf16> dA = A.to_device_async(stream);
  DeviceMatrix<bf16> dB = B.to_device_async(stream);
  DeviceMatrix<bf16> dC = C.to_device_async(stream);

  // warm up
  run_kernel<bf16>(args.kernel_type, dA, dB, dC, stream, args.block_size);
  cudaCheck(cudaStreamSynchronize(stream));

  // measure
  float matrixMulTimeMsec = 0;
  for (int i = 0; i < args.num_runs; i++) {
    float time =
        run_kernel<bf16>(args.kernel_type, dA, dB, dC, stream, args.block_size);
    matrixMulTimeMsec += time;
  }
  matrixMulTimeMsec /= args.num_runs;

  double flopsPerMatrixMul = 2.0 * static_cast<double>(args.width) *
                             static_cast<double>(args.height) *
                             static_cast<double>(args.width);
  double tFlops =
      (flopsPerMatrixMul * 1.0e-12f) / (matrixMulTimeMsec / 1000.0f);
  printf("Performance= %.2f TFLOP/s, Time= %.3f msec, Size= %.0f Ops,"
         " WorkgroupSize= %u threads/block\n",
         tFlops, matrixMulTimeMsec, flopsPerMatrixMul,
         args.block_size * args.block_size);

  printf("Copying results to host\n");
  C.from_device_async(dC, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  cudaCheck(cudaStreamDestroy(stream));

  printf("Verifying results\n");
  C.verify(C_ref);

  printf("Releasing memory\n");
  dC.close();
  dA.close();
  dB.close();

  A.close();
  B.close();
  C.close();

  return 0;
}