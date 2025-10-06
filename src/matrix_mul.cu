#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "utils.h"

typedef __nv_bfloat16 bf16;

static std::default_random_engine generator(786);

enum class KernelType { NAIVE, TILED };

__global__ void matrixMulNaiveKernel(DeviceMatrix<bf16> A, DeviceMatrix<bf16> B,
                                     DeviceMatrix<bf16> C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < C.height && col < C.width) {
    bf16 sum = 0.0;
    for (int k = 0; k < A.width; ++k) {
      sum += A.get_value(row, k) * B.get_value(k, col);
    }

    C.set_value(sum, row, col);
  }
}

template <int BLOCKSIZE>
__global__ void matrixMulTiledKernel(DeviceMatrix<bf16> A, DeviceMatrix<bf16> B,
                                     DeviceMatrix<bf16> C) {

  const int blockRow = blockIdx.y;
  const int row = threadIdx.y;
  const int blockCol = blockIdx.x;
  const int col = threadIdx.x;

  DeviceMatrix<bf16> Csub = C.get_block(blockRow, blockCol, BLOCKSIZE);
  bf16 sum = 0.0;
  for (int b = 0; b < (A.width / BLOCKSIZE); ++b) {
    __shared__ bf16 As[BLOCKSIZE][BLOCKSIZE];
    __shared__ bf16 Bs[BLOCKSIZE][BLOCKSIZE];

    DeviceMatrix<bf16> Asub = A.get_block(blockRow, b, BLOCKSIZE);
    DeviceMatrix<bf16> Bsub = B.get_block(b, blockCol, BLOCKSIZE);

    As[row][col] = Asub.get_value(row, col);
    Bs[row][col] = Bsub.get_value(row, col);

    __syncthreads();

    for (int k = 0; k < BLOCKSIZE; ++k) {
      sum += As[row][k] * Bs[k][col];
    }
  }
  Csub.set_value(sum, row, col);
}

template <int BLOCKSIZE>
float run_kernel(KernelType kernelType, DeviceMatrix<bf16> &dA,
                 DeviceMatrix<bf16> &dB, DeviceMatrix<bf16> &dC,
                 cudaStream_t stream) {
  cudaEvent_t startEvent, stopEvent;
  cudaCheck(cudaEventCreate(&startEvent));
  cudaCheck(cudaEventCreate(&stopEvent));

  dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
  dim3 dimGrid((dA.width + dimBlock.x - 1) / dimBlock.x,
               (dA.height + dimBlock.y - 1) / dimBlock.y);

  cudaCheck(cudaEventRecord(startEvent, 0));
  if (kernelType == KernelType::NAIVE) {
    matrixMulNaiveKernel<<<dimGrid, dimBlock, 0, stream>>>(dA, dB, dC);
  } else if (kernelType == KernelType::TILED) {
    matrixMulTiledKernel<BLOCKSIZE>
        <<<dimGrid, dimBlock, 0, stream>>>(dA, dB, dC);
  } else {
    throw std::runtime_error("Invalid kernel type");
  }
  cudaCheck(cudaEventRecord(stopEvent, 0));
  cudaCheck(cudaEventSynchronize(stopEvent));

  float gpuExecutionTime = 0;
  cudaCheck(cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent));
  return gpuExecutionTime;
}

int main() {
  // If WIDTH != HEIGHT, then A.width must be equal to B.height (k)
  const int WIDTH = 256;
  const int HEIGHT = 256;
  const int BLOCKSIZE = 32;
  const int NUM_RUNS = 100;

  HostMatrix<bf16> A(WIDTH, HEIGHT);
  HostMatrix<bf16> B(WIDTH, HEIGHT);
  HostMatrix<bf16> C(WIDTH, HEIGHT);

  A.randomize(generator);
  B.randomize(generator);

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  DeviceMatrix<bf16> dA = A.to_device_async(stream);
  DeviceMatrix<bf16> dB = B.to_device_async(stream);
  DeviceMatrix<bf16> dC = C.to_device_async(stream);

  // warm up
  run_kernel<BLOCKSIZE>(KernelType::NAIVE, dA, dB, dC, stream);
  run_kernel<BLOCKSIZE>(KernelType::TILED, dA, dB, dC, stream);

  float naiveTime = 0;
  float tiledTime = 0;

  // measure
  for (int i = 0; i < NUM_RUNS; i++) {
    float time = run_kernel<BLOCKSIZE>(KernelType::NAIVE, dA, dB, dC, stream);
    naiveTime += time;
  }
  naiveTime /= NUM_RUNS;
  for (int i = 0; i < NUM_RUNS; i++) {
    float time = run_kernel<BLOCKSIZE>(KernelType::TILED, dA, dB, dC, stream);
    tiledTime += time;
  }
  tiledTime /= NUM_RUNS;

  printf("Naive time: %f ms\n", naiveTime);
  printf("Tiled time: %f ms\n", tiledTime);

  printf("Copying results to host\n");
  C.from_device_async(dC, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  cudaCheck(cudaStreamDestroy(stream));

  C.verify(C); // TODO - replace C with a reference matrix
  dC.close();
  dA.close();
  dB.close();

  A.close();
  B.close();
  C.close();

  return 0;
}