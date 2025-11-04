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

enum class KernelType { Naive, Tiled, WmmaSimple };

std::string kernel_type_to_string(KernelType kernel_type) {
  static const std::map<KernelType, std::string> _kernel_type_to_string = {
      {KernelType::Naive, "Naive"},
      {KernelType::Tiled, "Tiled"},
      {KernelType::WmmaSimple, "WMMA Simple"},
  };

  return _kernel_type_to_string.at(kernel_type);
}

struct Args {
  int m;           // Matrix A is m x k
  int k;           // Matrix B is k x n
  int n;           // Matrix C is m x n
  int block_size;  // The Thread block size
  int tile_size;   // The Tile size for shared memory tiles or WMMA
};

int parse_args(int argc, char* argv[], Args& args, cudaDeviceProp& deviceProp) {
  argparse::ArgumentParser program("matrix_mul");
  std::string kernel_type;

  program.add_argument("--m").help("height of matrix A and C").scan<'i', int>().default_value(4096).store_into(args.m);

  program.add_argument("--n").help("width of matrix B and C").scan<'i', int>().default_value(4096).store_into(args.n);

  program.add_argument("--k")
      .help("width of matrix A and height of matrix B")
      .scan<'i', int>()
      .default_value(4096)
      .store_into(args.k);

  program.add_argument("--block-size")
      .help("thread block size")
      .scan<'i', int>()
      .default_value(16)
      .store_into(args.block_size);

  program.add_argument("--tile-size")
      .help("tile size for shared memory tiles or WMMA")
      .scan<'i', int>()
      .default_value(16)
      .store_into(args.tile_size);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if ((args.block_size * args.block_size) > deviceProp.maxThreadsPerBlock) {
    std::cerr << "Block size must be less than or equal to the maximum number "
                 "of threads per block"
              << std::endl;
    return 1;
  }

  if ((2 * args.tile_size * args.tile_size) > deviceProp.sharedMemPerBlock) {
    std::cerr << "Tile size must be less than or equal to the maximum number "
                 "of shared memory per tile"
              << std::endl;
    return 1;
  }

  printf("Arguments:\n");
  printf("Matrix A: %d x %d\n", args.m, args.k);
  printf("Matrix B: %d x %d\n", args.k, args.n);
  printf("Matrix C: %d x %d\n", args.m, args.n);
  printf("Thread block size: %d\n", args.block_size);
  printf("Tile size: %d\n", args.tile_size);

  return 0;
}


template <Numeric T>
__global__ void matrixMulNaiveKernel(T* A, T* B, T* C, int M, int K, int N) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    T sum = 0.0;
    for (int kk = 0; kk < K; ++kk) {
      sum += A[row * K + kk] * B[kk * N + col];
    }

    C[row * N + col] = sum;
  }
}

template <Numeric T>
__global__ void matrixMulTiledKernel(T* A, T* B, T* C, int M, int K, int N, int block_size) {
  const int blockRow = blockIdx.y;
  const int row = threadIdx.y;
  const int blockCol = blockIdx.x;
  const int col = threadIdx.x;

  extern __shared__ T smem[];
  T* As = &smem[0];
  T* Bs = &smem[block_size * block_size];

  T sum = 0.0;
  for (int b = 0; b < (K / block_size); ++b) {
    As[row * block_size + col] = A[(blockRow * block_size + row) * K + (b * block_size + col)];
    Bs[row * block_size + col] = B[(b * block_size + row) * K + (blockCol * block_size + col)];

    __syncthreads();

    for (int k = 0; k < block_size; ++k) {
      sum += As[row * block_size + k] * Bs[k * block_size + col];
    }

    __syncthreads();
  }
  C[(blockRow * block_size + row) * N + (blockCol * block_size + col)] = sum;
}

template <Numeric T, typename LayoutA = RowMajor, typename LayoutB = RowMajor, typename LayoutC = RowMajor>
void run_kernel(KernelType kernelType, Matrix<T, DeviceAsyncMemoryAllocator, LayoutA>& dA,
                Matrix<T, DeviceAsyncMemoryAllocator, LayoutB>& dB, Matrix<T, DeviceAsyncMemoryAllocator, LayoutC>& dC,
                Matrix<T, HostMemoryAllocator, LayoutC>& C, CudaStream& streamWrapper, Args& args) {
  cudaStream_t stream = streamWrapper.stream;
  CudaEventRecorder recorder = streamWrapper.record("matrix multiplication on GPU");
  int block_size = args.block_size;

  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid((dC.width() + dimBlock.x - 1) / dimBlock.x, (dC.height() + dimBlock.y - 1) / dimBlock.y);

  if (kernelType == KernelType::Naive) {
    matrixMulNaiveKernel<T><<<dimGrid, dimBlock, 0, stream>>>(dA.data, dB.data, dC.data, args.m, args.k, args.n);
  } else if (kernelType == KernelType::Tiled) {
    int shared_mem_size = 2 * block_size * block_size * sizeof(T);  // FIXME use tile size
    matrixMulTiledKernel<T>
        <<<dimGrid, dimBlock, shared_mem_size, stream>>>(dA.data, dB.data, dC.data, args.m, args.k, args.n, block_size);
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

  Matrix<bf16, HostMemoryAllocator> A(args.k, args.m);
  Matrix<bf16, HostMemoryAllocator> B(args.n, args.k);
  Matrix<bf16, HostMemoryAllocator> C1(args.n, args.m);  // change to float
  Matrix<bf16, HostMemoryAllocator> C2(args.n, args.m);

  A.randomize(generator);
  B.randomize(generator);

  CudaStream streamWrapper;
  cudaStream_t stream = streamWrapper.stream;

  Matrix<bf16, DeviceAsyncMemoryAllocator> dA(args.k, args.k, args.m, stream);
  Matrix<bf16, DeviceAsyncMemoryAllocator> dB(args.n, args.n, args.k, stream);
  Matrix<bf16, DeviceAsyncMemoryAllocator> dC(args.n, args.n, args.m, stream);

  cudaCheck(cudaMemcpyAsync(dA.data, A.data, A.size() * sizeof(bf16), cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemcpyAsync(dB.data, B.data, B.size() * sizeof(bf16), cudaMemcpyHostToDevice, stream));

  // Run naive and tiled kernels, assuming naive is correct, verify the results match
  run_kernel<bf16>(KernelType::Naive, dA, dB, dC, C1, streamWrapper, args);
  run_kernel<bf16>(KernelType::Tiled, dA, dB, dC, C2, streamWrapper, args);

  C1.verify(C2);

  return 0;
}