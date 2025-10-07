#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <stdlib.h>

#include "argparse.hpp"
#include "cuda_utils.h"

struct Args {
  int byte_count;
  int coarsening_factor;
  int block_size;
};

const int ONE_MB = 1 << 20;
const int NUM_BINS = 256;

int parse_args(int argc, char *argv[], Args &args, cudaDeviceProp &deviceProp) {
  argparse::ArgumentParser program("histogram");
  program.add_argument("--byte-count")
      .help("byte count in MB")
      .scan<'i', int>()
      .default_value(64)
      .store_into(args.byte_count);
  program.add_argument("--coarsening-factor")
      .help("coarsening factor")
      .scan<'i', int>()
      .default_value(1)
      .store_into(args.coarsening_factor);
  program.add_argument("--block-size")
      .help("block size")
      .scan<'i', int>()
      .default_value(256)
      .store_into(args.block_size);
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
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
unsigned int *getHistogramCpu(unsigned char *inputData, int byte_count) {
  unsigned int *hHistogramCPU = new unsigned int[NUM_BINS];
  memset(hHistogramCPU, 0, NUM_BINS * sizeof(unsigned int));

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < byte_count; i++) {
    hHistogramCPU[inputData[i]]++;
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpuDuration = (stop - start);
  printf("Time to calculate histogram on CPU: %f ms\n", cpuDuration.count());
  return hHistogramCPU;
}

// Allocate the memory and generate the input data using a uniform distribution
// from 0 to 255
unsigned char *generateInputData(int byte_count) {
  unsigned char *inputData = new unsigned char[byte_count];
  std::default_random_engine generator(786);
  std::uniform_int_distribution<unsigned char> distribution(0, 255);
  for (int i = 0; i < byte_count; i++) {
    inputData[i] = distribution(generator);
  }
  return inputData;
}

bool verifyHistogram(unsigned int *histogramCPU, unsigned int *histogramGPU) {
  for (int i = 0; i < NUM_BINS; i++) {
    if (histogramCPU[i] != histogramGPU[i]) {
      printf("Histogram mismatch at index %d: %d != %d\n", i, histogramCPU[i],
             histogramGPU[i]);
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
__global__ void histogramKernel(unsigned char *d_inputData,
                                unsigned int *d_histogramGPU, int byte_count,
                                int coarsening_factor) {

  // Use a private histogram per block, allocate it in shared memory to be of
  // NUM_BINS size and make block threads cooperatively initiliaze it to 0
  __shared__ unsigned int s_histogram[NUM_BINS];
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    s_histogram[i] = 0;
  }
  __syncthreads();

  // Each thread is responsible for updating coarsening_factor input elements
  // into the private histogram
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx * coarsening_factor; i < (idx + 1) * coarsening_factor;
       ++i) {
    if (i < byte_count) {
      atomicAdd(&s_histogram[d_inputData[i]], 1);
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

void launchKernel(unsigned char *d_inputData, unsigned int *d_histogramGPU,
                  const Args &args, cudaStream_t stream) {
  dim3 dimBlock(args.block_size);
  // The number of items processed by one block is the size of the block times
  // the coarsening factor because each thread is responsible for coarsening
  // factor number of items.
  int dimBlockTotal = dimBlock.x * args.coarsening_factor;
  dim3 dimGrid((args.byte_count + dimBlockTotal - 1) / dimBlockTotal);
  histogramKernel<<<dimGrid, dimBlock, 0, stream>>>(
      d_inputData, d_histogramGPU, args.byte_count, args.coarsening_factor);
  cudaCheck(cudaGetLastError());
}

int main(int argc, char *argv[]) {
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  Args args;
  if (parse_args(argc, argv, args, deviceProp) != 0) {
    return 1;
  }

  // Allocate host memory and perform initializations and CPU calculations
  unsigned char *inputData = generateInputData(args.byte_count);
  unsigned int *histogramCPU = getHistogramCpu(inputData, args.byte_count);
  unsigned int *histogramGPU = new unsigned int[NUM_BINS];
  cudaCheck(cudaHostRegister((void *)inputData, args.byte_count,
                             cudaHostRegisterDefault));

  // Create a stream, the events, and allocate device memory
  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t startEvent, stopEvent;
  cudaCheck(cudaEventCreate(&startEvent));
  cudaCheck(cudaEventCreate(&stopEvent));
  unsigned char *d_inputData = nullptr;
  unsigned int *d_histogramGPU = nullptr;
  cudaCheck(cudaMallocAsync((void **)&d_inputData, args.byte_count, stream));
  cudaCheck(cudaMallocAsync((void **)&d_histogramGPU, NUM_BINS * sizeof(uint),
                            stream));
  printf("Calculate histogram on GPU\n");

  cudaCheck(cudaEventRecord(startEvent, stream));
  cudaCheck(cudaMemcpyAsync(d_inputData, inputData, args.byte_count,
                            cudaMemcpyHostToDevice, stream));
  launchKernel(d_inputData, d_histogramGPU, args, stream);

  cudaCheck(cudaMemcpyAsync(histogramGPU, d_histogramGPU,
                            NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost,
                            stream));
  cudaCheck(cudaEventRecord(stopEvent, stream));
  cudaCheck(cudaEventSynchronize(stopEvent));
  cudaCheck(cudaStreamSynchronize(stream));
  float gpuExecutionTime = 0;
  cudaCheck(cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent));
  printf("Time to calculate histogram on GPU: %f ms, throuput %f MB/s, size %d "
         "MB\n",
         gpuExecutionTime, 1e-06 * args.byte_count / gpuExecutionTime,
         args.byte_count / ONE_MB);

  bool isVerified = verifyHistogram(histogramCPU, histogramGPU);

  cudaCheck(cudaFreeAsync(d_inputData, stream));
  cudaCheck(cudaFreeAsync(d_histogramGPU, stream));

  cudaCheck(cudaEventDestroy(startEvent));
  cudaCheck(cudaEventDestroy(stopEvent));
  cudaCheck(cudaStreamDestroy(stream));

  delete[] inputData;
  delete[] histogramCPU;
  delete[] histogramGPU;

  return isVerified ? 0 : 1;
}