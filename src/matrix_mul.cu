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

__global__ void matrixMulNaiveKernel(bf16 *A, bf16 *B, bf16 *C, int WIDTH,
                                     int HEIGHT) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < WIDTH && col < HEIGHT) {
    bf16 sum = 0.0;
    for (int k = 0; k < WIDTH; ++k) {
      sum += A[row * WIDTH + k] * B[k * WIDTH + col];
    }

    C[row * WIDTH + col] = sum;
  }
}

int main() {
  // If WIDTH != HEIGHT, then A.width must be equal to B.height (k)
  const int WIDTH = 256;
  const int HEIGHT = 256;
  const int BLOCKSIZE = 32;

  Matrix<bf16> A(WIDTH, HEIGHT, MatrixType::HOST);
  Matrix<bf16> B(WIDTH, HEIGHT, MatrixType::HOST);
  Matrix<bf16> C(WIDTH, HEIGHT, MatrixType::HOST);

  A.randomize(generator);
  B.randomize(generator);

  Matrix<bf16> dA(WIDTH, HEIGHT, MatrixType::DEVICE);
  Matrix<bf16> dB(WIDTH, HEIGHT, MatrixType::DEVICE);
  Matrix<bf16> dC(WIDTH, HEIGHT, MatrixType::DEVICE);

  dA = A;
  dB = B;

  dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
  dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
               (HEIGHT + blockDim.y - 1) / blockDim.y);
  printf("Kernel started\n");
  matrixMulNaiveKernel<<<gridDim, blockDim>>>(dA.get_data(), dB.get_data(),
                                              dC.get_data(), WIDTH, HEIGHT);
  printf("Kernel ended\n");
  C = dC;
  cudaCheck(cudaDeviceSynchronize());

  C.verify(C); // TODO - replace C with a reference matrix
  return 0;
}