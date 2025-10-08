#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <random>
#include <vector>

#include "argparse.hpp"
#include "cuda_utils.h"

struct Args {
  int size;
  int block_size;
};

int parse_args(int argc, char* argv[], Args& args, cudaDeviceProp& deviceProp) {
  argparse::ArgumentParser program("histogram");
  program.add_argument("--size")
      .help("size of the array to reduce")
      .scan<'i', int>()
      .default_value(1 << 24)
      .store_into(args.size);
  program.add_argument("--block-size")
      .help("block size")
      .scan<'i', int>()
      .default_value(128)
      .store_into(args.block_size);
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  printf("Arguments:\n");
  printf("  Size: %d\n", args.size);
  printf("  Block size: %d\n", args.block_size);
  return 0;
}

std::vector<int> generateInputData(int size) {
  std::vector<int> inputData(size);
  std::default_random_engine generator(786);
  std::uniform_int_distribution<int> distribution(0, 100);
  for (int i = 0; i < size; i++) {
    inputData[i] = distribution(generator);
  }
  return inputData;
}

unsigned long long reduceCpu(const std::vector<int>& inputData) {
  auto start = std::chrono::high_resolution_clock::now();
  unsigned long long sum = 0;
  for (int i = 0; i < inputData.size(); i++) {
    sum += inputData[i];
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpuDuration = (stop - start);
  printf("Time to reduce on CPU: %f ms\n", cpuDuration.count());
  return sum;
}

__global__ void reductionKernelNaive(int* d_inputData, unsigned long long* d_outputData, int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    atomicAdd(d_outputData, d_inputData[tid]);
  }
}

void runKernel(int* d_inputData, unsigned long long* d_outputData, Args& args) {
  dim3 dimBlock(args.block_size);
  dim3 dimGrid((args.size + args.block_size - 1) / args.block_size);
  reductionKernelNaive<<<dimGrid, dimBlock>>>(d_inputData, d_outputData, args.size);
}

int main(int argc, char* argv[]) {
  Args args;
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  if (parse_args(argc, argv, args, deviceProp) != 0) {
    return 1;
  }

  const std::vector<int> inputData = generateInputData(args.size);
  const unsigned long long cpuResult = reduceCpu(inputData);
  unsigned long long gpuResult = 0;

  cudaCheck(cudaHostRegister((void*)inputData.data(), args.size * sizeof(int), cudaHostRegisterDefault));
  int* d_inputData = nullptr;
  cudaCheck(cudaMalloc((void**)&d_inputData, args.size * sizeof(int)));
  cudaCheck(cudaMemcpy(d_inputData, inputData.data(), args.size * sizeof(int), cudaMemcpyHostToDevice));
  unsigned long long* d_outputData = nullptr;
  cudaCheck(cudaMalloc((void**)&d_outputData, sizeof(unsigned long long)));

  cudaEvent_t startEvent, stopEvent;
  cudaCheck(cudaEventCreate(&startEvent));
  cudaCheck(cudaEventCreate(&stopEvent));
  cudaCheck(cudaEventRecord(startEvent, 0));
  runKernel(d_inputData, d_outputData, args);
  cudaCheck(cudaEventRecord(stopEvent, 0));
  cudaCheck(cudaEventSynchronize(stopEvent));
  float gpuExecutionTime = 0;
  cudaCheck(cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent));
  printf("Time to execute on GPU: %f ms\n", gpuExecutionTime);
  cudaCheck(cudaDeviceSynchronize());

  cudaCheck(cudaMemcpy(&gpuResult, d_outputData, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  printf("GPU result: %llu\n", gpuResult);
  printf("CPU result: %llu\n", cpuResult);
  printf("Difference: %llu\n", gpuResult - cpuResult);

  return 0;
}