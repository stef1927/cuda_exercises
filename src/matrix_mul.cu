#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "utils.h"

typedef __nv_bfloat16 bf16;

std::default_random_engine generator(786);

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

int main() {
  // If WIDTH != HEIGHT, then A.width must be equal to B.height (k)
  const int WIDTH = 256;
  const int HEIGHT = 256;
  const int BLOCKSIZE = 32;
  const bool NAIVE = false;

  HostMatrix<bf16> A(WIDTH, HEIGHT);
  HostMatrix<bf16> B(WIDTH, HEIGHT);
  HostMatrix<bf16> C(WIDTH, HEIGHT);

  A.randomize(generator);
  B.randomize(generator);

  DeviceMatrix<bf16> dA = A.to_device();
  DeviceMatrix<bf16> dB = B.to_device();
  DeviceMatrix<bf16> dC = C.to_device();

  dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
  dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x,
               (HEIGHT + dimBlock.y - 1) / dimBlock.y);

  printf("Kernel started: %s\n", NAIVE ? "naive" : "tiled");
  if (NAIVE) {
    matrixMulNaiveKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);
  } else {
    matrixMulTiledKernel<BLOCKSIZE><<<dimGrid, dimBlock>>>(dA, dB, dC);
  }
  printf("Kernel ended\n");
  cudaCheck(cudaDeviceSynchronize());

  printf("Copying results to host\n");
  C.from_device(dC);
  cudaCheck(cudaDeviceSynchronize());

  C.verify(C); // TODO - replace C with a reference matrix
  dC.close();
  dA.close();
  dB.close();

  A.close();
  B.close();
  C.close();

  return 0;
}