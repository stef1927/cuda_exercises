#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "argparse.hpp"
#include "cpp_utils.hpp"
#include "cuda_utils.cuh"
#include "matrix_mul_gpu.cuh"

typedef __nv_bfloat16 bf16;

static std::default_random_engine generator(786);

enum class KernelType { NAIVE, TILED };

std::string kernel_type_to_string(KernelType kernel_type) {
  static const std::map<KernelType, std::string> _kernel_type_to_string = {
      {KernelType::NAIVE, "Naive"},
      {KernelType::TILED, "Tiled"},
  };

  return _kernel_type_to_string.at(kernel_type);
}

struct Args {
  int width;
  int height;
  int block_size;
};

int parse_args(int argc, char* argv[], Args& args, cudaDeviceProp& deviceProp) {
  argparse::ArgumentParser program("matrix_mul");
  std::string kernel_type;

  program.add_argument("--width")
      .help("width of the matrix")
      .scan<'i', int>()
      .default_value(4096)
      .store_into(args.width);

  program.add_argument("--height")
      .help("height of the matrix")
      .scan<'i', int>()
      .default_value(4096)
      .store_into(args.height);

  program.add_argument("--block-size")
      .help("block size")
      .scan<'i', int>()
      .default_value(16)
      .store_into(args.block_size);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  // current limitation, for later on, we need to support different width and
  // height for A and B and A.width must be equal to B.height (k)
  if (args.width != args.height) {
    std::cerr << "Width and height must be equal" << std::endl;
    return 1;
  }

  if ((args.block_size * args.block_size) > deviceProp.maxThreadsPerBlock) {
    std::cerr << "Block size must be less than or equal to the maximum number "
                 "of threads per block"
              << std::endl;
    return 1;
  }

  if ((2 * args.block_size * args.block_size) > deviceProp.sharedMemPerBlock) {
    std::cerr << "Block size must be less than or equal to the maximum number "
                 "of shared memory per block"
              << std::endl;
    return 1;
  }

  printf("Arguments:\n");
  printf("Width: %d\n", args.width);
  printf("Height: %d\n", args.height);
  printf("Block size: %d\n", args.block_size);

  return 0;
}


template <Numeric T, typename LayoutA = RowMajor, typename LayoutB = RowMajor, typename LayoutC = RowMajor>
__global__ void matrixMulNaiveKernel(Matrix<T, NoAllocator, LayoutA> A, Matrix<T, NoAllocator, LayoutB> B,
                                     Matrix<T, NoAllocator, LayoutC> C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < C.height() && col < C.width()) {
    T sum = 0.0;
    for (int k = 0; k < A.width(); ++k) {
      sum += A(row, k) * B(k, col);
    }

    C(row, col) = sum;
  }
}

template <Numeric T, typename LayoutA = RowMajor, typename LayoutB = RowMajor, typename LayoutC = RowMajor>
__global__ void matrixMulTiledKernel(Matrix<T, NoAllocator, LayoutA> A, Matrix<T, NoAllocator, LayoutB> B,
                                     Matrix<T, NoAllocator, LayoutC> C, int block_size) {
  const int blockRow = blockIdx.y;
  const int row = threadIdx.y;
  const int blockCol = blockIdx.x;
  const int col = threadIdx.x;

  extern __shared__ T smem[];
  T* As = &smem[0];
  T* Bs = &smem[block_size * block_size];

  Matrix<T, NoAllocator, LayoutC> Csub = C.get_block(blockRow, blockCol, block_size);
  T sum = 0.0;
  for (int b = 0; b < (A.width() / block_size); ++b) {
    Matrix<T, NoAllocator, LayoutA> Asub = A.get_block(blockRow, b, block_size);
    Matrix<T, NoAllocator, LayoutB> Bsub = B.get_block(b, blockCol, block_size);

    As[row * block_size + col] = Asub(row, col);
    Bs[row * block_size + col] = Bsub(row, col);

    __syncthreads();

    for (int k = 0; k < block_size; ++k) {
      sum += As[row * block_size + k] * Bs[k * block_size + col];
    }

    __syncthreads();
  }
  Csub(row, col) = sum;
}

template <Numeric T, typename LayoutA = RowMajor, typename LayoutB = RowMajor, typename LayoutC = RowMajor>
void run_kernel(KernelType kernelType, Matrix<T, DeviceAsyncMemoryAllocator, LayoutA>& dA,
                Matrix<T, DeviceAsyncMemoryAllocator, LayoutB>& dB, Matrix<T, DeviceAsyncMemoryAllocator, LayoutC>& dC,
                Matrix<T, HostMemoryAllocator, LayoutC>& C, CudaStream& streamWrapper, int block_size) {
  cudaStream_t stream = streamWrapper.stream;
  CudaEventRecorder recorder = streamWrapper.record("matrix multiplication on GPU");

  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid((dA.width() + dimBlock.x - 1) / dimBlock.x, (dB.height() + dimBlock.y - 1) / dimBlock.y);

  if (kernelType == KernelType::NAIVE) {
    matrixMulNaiveKernel<T><<<dimGrid, dimBlock, 0, stream>>>(dA.copy(), dB.copy(), dC.copy());
  } else if (kernelType == KernelType::TILED) {
    int shared_mem_size = 2 * block_size * block_size * sizeof(T);
    matrixMulTiledKernel<T>
        <<<dimGrid, dimBlock, shared_mem_size, stream>>>(dA.copy(), dB.copy(), dC.copy(), block_size);
  } else {
    throw std::runtime_error("Invalid kernel type");
  }
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaStreamSynchronize(stream));

  double matrixMulTimeMsec = recorder.close();
  double flopsPerMatrixMul =
      2.0 * static_cast<double>(dA.width()) * static_cast<double>(dB.height()) * static_cast<double>(dA.width());
  double tFlops = (flopsPerMatrixMul * 1.0e-12f) / (matrixMulTimeMsec / 1000.0f);
  printf(
      "%s/%s: Performance= %.2f TFLOP/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      dB.layout.name, kernel_type_to_string(kernelType).c_str(), tFlops, matrixMulTimeMsec, flopsPerMatrixMul,
      block_size * block_size);

  cudaCheck(cudaMemcpyAsync(C.data, dC.data, C.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaStreamSynchronize(stream));
}

int main(int argc, char* argv[]) {
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  Args args;
  if (parse_args(argc, argv, args, deviceProp) != 0) {
    return 1;
  }

  Matrix<bf16, HostMemoryAllocator> A(args.width, args.height);
  Matrix<bf16, HostMemoryAllocator> B(args.width, args.height);
  Matrix<bf16, HostMemoryAllocator> C1(args.width, args.height);
  Matrix<bf16, HostMemoryAllocator> C2(args.width, args.height);

  A.randomize(generator);
  B.randomize(generator);

  CudaStream streamWrapper;
  cudaStream_t stream = streamWrapper.stream;

  Matrix<bf16, DeviceAsyncMemoryAllocator> dA(args.width, args.width, args.height, stream);
  Matrix<bf16, DeviceAsyncMemoryAllocator> dB(args.width, args.width, args.height, stream);
  Matrix<bf16, DeviceAsyncMemoryAllocator> dC(args.width, args.width, args.height, stream);

  cudaCheck(cudaMemcpyAsync(dA.data, A.data, A.size() * sizeof(bf16), cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemcpyAsync(dB.data, B.data, B.size() * sizeof(bf16), cudaMemcpyHostToDevice, stream));

  // Run naive and tiled kernels, assuming naive is correct, verify the results match
  run_kernel<bf16>(KernelType::NAIVE, dA, dB, dC, C1, streamWrapper, args.block_size);
  run_kernel<bf16>(KernelType::TILED, dA, dB, dC, C2, streamWrapper, args.block_size);

  C1.verify(C2);

  return 0;
}