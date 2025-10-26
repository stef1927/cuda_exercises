#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <random>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.hpp"
#include "cuda_utils.cuh"

namespace cg = cooperative_groups;

struct Args {
  int size;
  int block_size;
  int kernel_number;
};

inline int nextPow2(int n) {
  if (n == 0)
    return 1;
  return 1 << (32 - __builtin_clz(n - 1));
}

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
      .default_value(256)
      .store_into(args.block_size);
  program.add_argument("--kernel-number")
      .help("kernel number")
      .scan<'i', int>()
      .default_value(2)
      .store_into(args.kernel_number);
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if (args.block_size < 32) {
    printf("Setting block size to 32\n");
    args.block_size = 32;
  }

  printf("Setting block size to next power of 2\n");
  args.block_size = nextPow2(args.block_size);

  printf("Arguments:\n");
  printf("  Size: %d\n", args.size);
  printf("  Block size: %d\n", args.block_size);
  printf("  Kernel number: %d\n", args.kernel_number);
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
  Timer timer("reduce on the CPU");
  unsigned long long sum = 0;
  for (int i = 0; i < inputData.size(); i++) {
    sum += inputData[i];
  }
  return sum;
}

// Naive kernel where all threads perform an atomic add, this will not perform well because
// it will have a lot of contention on the atomic add
__global__ void reductionKernel0(int* d_inputData, int size, unsigned long long* d_outputData) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    atomicAdd(d_outputData, d_inputData[tid]);
  }
}

// In this kernel, each block reduces twice the block size from the input data, each thread reads
// its natural element and the element one block apart from global to shared memory. Then the
// stride is progressively halved until only one result remains, the one for the first thread,
// which is then written to the output. The cooperative threads API are used to synchronize
// every pass into shared memory.
__global__ void reductionKernel1(int* d_inputData, int size, unsigned long long* d_outputData) {
  auto cta = cg::this_thread_block();      // the thread block group
  extern __shared__ unsigned int sdata[];  // blockDim.x * sizeof(unsigned int)

  // Reduce from global memory to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  unsigned int sum = (i < size) ? d_inputData[i] : 0;

  if (i + blockDim.x < size)
    sum += d_inputData[i + blockDim.x];

  sdata[tid] = sum;
  cta.sync();

  // Then reduce from shared memory
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = sum = sum + sdata[tid + stride];
    }

    cta.sync();
  }

  // The first thread in the block writes the result to global memory
  if (tid == 0) {
    atomicAdd(d_outputData, sum);
  }
}

/// In this kernel, we use a cooperative thread block and thread tiles of size 32 (warps) to reduce
/// the input data using hardware accelerated reduction for warp tiles. Each thread computes its own
/// sum from global memory by adding values over the stride and then each thread sum is reduced across
/// tiles using the CG reduce() function.
__global__ void reductionKernel2(int* d_inputData, int size, unsigned long long* d_outputData) {
  auto cta = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(cta);
  int tid = cta.group_index().x * (cta.group_dim().x * 2) + cta.thread_index().x;
  unsigned int thread_sum = (tid < size) ? d_inputData[tid] : 0;

  if (tid + cta.group_dim().x < size) {
    thread_sum += d_inputData[tid + cta.group_dim().x];
  }

  // reduce thread sums across each warp, cg::plus<int> allows cg::reduce() to
  // know it can use hardware acceleration for addition
  unsigned int warp_sum = cg::reduce(warp, thread_sum, cg::plus<unsigned int>());

  if (warp.thread_rank() == 0) {
    atomicAdd(d_outputData, (unsigned long long)warp_sum);
  }
}

void runKernel(int* d_inputData, unsigned long long* d_outputData, Args& args) {
  switch (args.kernel_number) {
    case 0: {
      dim3 dimBlock(args.block_size);
      dim3 dimGrid((args.size + args.block_size - 1) / args.block_size);
      reductionKernel0<<<dimGrid, dimBlock, 0>>>(d_inputData, args.size, d_outputData);
      break;
    }
    case 1: {
      dim3 dimBlock(args.block_size);
      dim3 dimGrid((args.size + args.block_size * 2 - 1) / (args.block_size * 2));
      unsigned int shared_mem_size = dimBlock.x * sizeof(unsigned int);
      reductionKernel1<<<dimGrid, dimBlock, shared_mem_size>>>(d_inputData, args.size, d_outputData);
      break;
    }
    case 2: {
      dim3 dimBlock(args.block_size);
      dim3 dimGrid((args.size + args.block_size * 2 - 1) / (args.block_size * 2));
      reductionKernel2<<<dimGrid, dimBlock>>>(d_inputData, args.size, d_outputData);
      break;
    }
    default:
      throw std::runtime_error("Invalid kernel number: " + std::to_string(args.kernel_number));
  }
  cudaCheck(cudaGetLastError());
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

  {
    Timer timer("reduce on the GPU");
    auto d_inputData = make_cuda_unique<int>(args.size);
    auto d_outputData = make_cuda_unique<unsigned long long>(1);
    cudaCheck(cudaHostRegister((void*)inputData.data(), args.size * sizeof(int), cudaHostRegisterDefault));
    cudaCheck(cudaMemcpy(d_inputData.get(), inputData.data(), args.size * sizeof(int), cudaMemcpyHostToDevice));

    {
      CudaEventRecorder recorder("reduce on the GPU");
      runKernel(d_inputData.get(), d_outputData.get(), args);
      cudaCheck(cudaDeviceSynchronize());
    }

    cudaCheck(cudaMemcpy(&gpuResult, d_outputData.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  }
  printf("GPU result: %llu\n", gpuResult);
  printf("CPU result: %llu\n", cpuResult);
  printf("Difference: %llu\n", gpuResult > cpuResult ? gpuResult - cpuResult : cpuResult - gpuResult);

  return 0;
}