#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <memory>
#include <random>

#include "argparse.hpp"
#include "cpp_utils.h"
#include "cuda_utils.h"

struct Args {
  int byte_count;
  int coarsening_factor;
  int block_size;
};

const int ONE_MB = 1 << 20;
const int NUM_BINS = 256;

int parse_args(int argc, char* argv[], Args& args, cudaDeviceProp& deviceProp) {
  argparse::ArgumentParser program("histogram");
  program.add_argument("--byte-count")
      .help("byte count in MB")
      .scan<'i', int>()
      .default_value(128)
      .store_into(args.byte_count);
  program.add_argument("--coarsening-factor")
      .help("coarsening factor")
      .scan<'i', int>()
      .default_value(4)
      .store_into(args.coarsening_factor);
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

  args.byte_count *= ONE_MB;

  printf("Arguments:\n");
  printf("  Byte count: %d MB\n", args.byte_count / ONE_MB);
  printf("  Coarsening factor: %d\n", args.coarsening_factor);
  return 0;
}

// Allocate the memory and calculates the histogram on the CPU as a reference
// check for correctness
std::unique_ptr<unsigned int[]> getHistogramCpu(const std::unique_ptr<unsigned char[]>& inputData, int byte_count) {
  Timer timer("calculate histogram on CPU");
  std::unique_ptr<unsigned int[]> hHistogramCPU = std::make_unique<unsigned int[]>(NUM_BINS);
  memset(hHistogramCPU.get(), 0, NUM_BINS * sizeof(unsigned int));

  for (int i = 0; i < byte_count; i++) {
    hHistogramCPU[inputData[i]]++;
  }

  return hHistogramCPU;
}

// Allocate the memory and generate the input data using a uniform distribution
// from 0 to 255
std::unique_ptr<unsigned char[]> generateInputData(int byte_count) {
  std::unique_ptr<unsigned char[]> inputData = std::make_unique<unsigned char[]>(byte_count);
  std::default_random_engine generator(786);
  std::uniform_int_distribution<unsigned char> distribution(0, 255);
  for (int i = 0; i < byte_count; i++) {
    inputData[i] = distribution(generator);
  }
  return inputData;
}

bool verifyHistogram(const std::unique_ptr<unsigned int[]>& histogramCPU,
                     const std::unique_ptr<unsigned int[]>& histogramGPU) {
  for (int i = 0; i < NUM_BINS; i++) {
    if (histogramCPU[i] != histogramGPU[i]) {
      printf("Histogram mismatch at index %d: should: %d, is: %d\n", i, histogramCPU[i], histogramGPU[i]);
      return false;
    }
  }
  printf("Histogram verified successfully\n");
  return true;
}

// Kernel that demonstrates the use of shared memory to privatize the histogram
// with one block owning a private histogram and merging it into the global
// histogram. The coarsening factor is the number of input elements that one
// thread is responsible for updating in the private histogram.
__global__ void histogramKernel(const unsigned int* d_inputData, int input_size, unsigned int* d_histogramGPU,
                                int coarsening_factor) {
  // Use a private histogram per block, allocate it in shared memory to be of
  // NUM_BINS size and make block threads cooperatively initiliaze it to 0
  __shared__ unsigned int s_histogram[NUM_BINS];
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    s_histogram[i] = 0;
  }
  __syncthreads();

  // Each thread is responsible for updating coarsening_factor input elements
  // into the private histogram, we skip by a block to keep memory access coalesced
  int start_idx = (blockIdx.x * blockDim.x) * coarsening_factor + threadIdx.x;
  for (int i = 0; i < coarsening_factor; ++i) {
    int idx = start_idx + (i * blockDim.x);
    if (idx < input_size) {
      unsigned int data = d_inputData[idx];
      atomicAdd(&s_histogram[(data >> 0) & 0xFFU], 1);
      atomicAdd(&s_histogram[(data >> 8) & 0xFFU], 1);
      atomicAdd(&s_histogram[(data >> 16) & 0xFFU], 1);
      atomicAdd(&s_histogram[(data >> 24) & 0xFFU], 1);
    }
  }
  __syncthreads();

  // Merge the private histogram into the global histogram
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    unsigned int val = s_histogram[i];
    atomicAdd(&d_histogramGPU[i], val);
  }
  __syncthreads();
}

void launchKernel(unsigned char* d_inputData, unsigned int* d_histogramGPU, const Args& args, cudaStream_t stream) {
  // The kernel will load input data as unsigned ints to reduce global memory
  // access requests and ensure that they are aligned.
  if (args.byte_count % sizeof(unsigned int) != 0) {
    throw std::runtime_error("Byte count must be divisible by " + std::to_string(sizeof(unsigned int)));
  }
  int input_size = args.byte_count / sizeof(unsigned int);
  dim3 dimBlock(args.block_size);
  // The number of items processed by one block is the size of the block times
  // the coarsening factor because each thread is responsible for coarsening
  // factor number of items.
  int dimBlockTotal = dimBlock.x * args.coarsening_factor;
  dim3 dimGrid((input_size + dimBlockTotal - 1) / dimBlockTotal);
  printf("input_size: %d, dimBlock: %d, dimBlockTotal: %d, dimGrid: %d\n", input_size, dimBlock.x, dimBlockTotal,
         dimGrid.x);

  histogramKernel<<<dimGrid, dimBlock, 0, stream>>>((unsigned int*)d_inputData, input_size, d_histogramGPU,
                                                    args.coarsening_factor);
  cudaCheck(cudaGetLastError());
}

int main(int argc, char* argv[]) {
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  Args args;
  if (parse_args(argc, argv, args, deviceProp) != 0) {
    return 1;
  }

  // Allocate host memory and perform initializations and CPU calculations
  auto inputData = generateInputData(args.byte_count);
  auto histogramCPU = getHistogramCpu(inputData, args.byte_count);
  auto histogramGPU = std::make_unique<unsigned int[]>(NUM_BINS);
  cudaCheck(cudaHostRegister((void*)inputData.get(), args.byte_count * sizeof(unsigned char), cudaHostRegisterDefault));

  CudaStream streamWrapper;
  cudaStream_t stream = streamWrapper.stream;

  auto d_inputData = make_cuda_unique<unsigned char>(args.byte_count);
  auto d_histogramGPU = make_cuda_unique<unsigned int>(NUM_BINS);
  {
    Timer timer("calculate histogram on GPU");
    cudaCheck(cudaMemcpyAsync(d_inputData.get(), inputData.get(), args.byte_count, cudaMemcpyHostToDevice, stream));

    {
      CudaEventRecorder recorder = streamWrapper.record("calculate histogram on GPU");
      printf("Calculate histogram on GPU\n");
      launchKernel(d_inputData.get(), d_histogramGPU.get(), args, stream);
    }

    cudaCheck(cudaMemcpyAsync(histogramGPU.get(), d_histogramGPU.get(), NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost,
                              stream));
    cudaCheck(cudaStreamSynchronize(stream));
  }
  return verifyHistogram(histogramCPU, histogramGPU) ? 0 : 1;
}