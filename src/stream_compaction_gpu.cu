#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cub/cub.cuh>  // or equivalently <cub/device/device_scan.cuh>
#include <random>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.hpp"
#include "cuda_utils.cuh"
#include "stream_compaction_utils.hpp"


struct Args {
  int size;
  int block_size;  // GPU block size
  bool debug_print;
};


int parse_args(int argc, char* argv[], Args& args, cudaDeviceProp& deviceProp) {
  argparse::ArgumentParser program("stream_compaction_gpu");
  program.add_argument("--size")
      .help("The size of the array to scan")
      .scan<'i', int>()
      .default_value(1 << 24)
      .store_into(args.size);
  program.add_argument("--block-size")
      .help("The block size")
      .scan<'i', int>()
      .default_value(1024)
      .store_into(args.block_size);
  program.add_argument("--debug-print")
      .help("Whether to print debug information")
      .default_value(false)
      .store_into(args.debug_print);
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


// Generate input data with CUDA unified memory
CudaUniquePtr<int> generate_input_data_cuda(int size) {
  NVTXScopedRange fn("generate_input_data_cuda");
  auto input_data = make_cuda_unique<int>(size, true);
  std::default_random_engine generator(786);
  std::uniform_int_distribution<int> distribution(0, 100);
  for (int i = 0; i < size; i++) {
    input_data.get()[i] = distribution(generator);
  }
  return input_data;
}


template <typename Predicate>
__global__ void create_input_data_indicators_kernel(int* input_data, int* input_data_indicators, size_t size,
                                                    Predicate predicate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    input_data_indicators[tid] = predicate(input_data[tid]) ? 1 : 0;
  }
}

template <typename Predicate>
void create_input_data_indicators(CudaStream& streamWrapper, int* input_data, int* input_data_indicators, size_t size,
                                  int block_size, Predicate predicate) {
  auto recorder = streamWrapper.record("create_input_data_indicators_kernel");
  cudaStream_t stream = streamWrapper.stream;
  dim3 dimBlock(block_size);
  dim3 dimGrid((size + block_size - 1) / block_size);

  create_input_data_indicators_kernel<<<dimGrid, dimBlock, 0, stream>>>(input_data, input_data_indicators, size,
                                                                        predicate);
  cudaCheck(cudaGetLastError());
}

void cub_inclusive_scan(CudaStream& streamWrapper, CudaUniquePtr<int>& d_input_data, CudaUniquePtr<int>& d_output_data,
                        int size) {
  cudaStream_t stream = streamWrapper.stream;
  CudaEventRecorder recorder = streamWrapper.record("inclusive scan on the GPU using the CUB library");

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cudaCheck(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_input_data.get(), d_output_data.get(), size,
                                          stream));
  cudaCheck(cudaGetLastError());

  // Allocate temporary storage
  auto d_temp_storage = make_cuda_unique<char>(temp_storage_bytes);

  // Run exclusive prefix sum
  cudaCheck(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_input_data.get(),
                                          d_output_data.get(), size, stream));
  cudaCheck(cudaGetLastError());
}


template <typename Predicate>
__global__ void create_output_data_kernel(int* input_data, int* input_data_indicators, int* output_data,
                                          size_t input_size, Predicate predicate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < input_size && predicate(input_data[tid])) {
    output_data[input_data_indicators[tid] - 1] = input_data[tid];
  }
}

template <typename Predicate>
void create_output_data(CudaStream& streamWrapper, int* input_data, int* input_data_indicators, int* output_data,
                        size_t input_size, int block_size, Predicate predicate) {
  auto recorder = streamWrapper.record("create_output_data_kernel");
  cudaStream_t stream = streamWrapper.stream;
  dim3 dimBlock(block_size);
  dim3 dimGrid((input_size + block_size - 1) / block_size);

  create_output_data_kernel<<<dimGrid, dimBlock, 0, stream>>>(input_data, input_data_indicators, output_data,
                                                              input_size, predicate);
  cudaCheck(cudaGetLastError());
}

// This function performs stream compaction on the GPU using a 3-pass approach:
// - The first pass runs a kernel that marks the output data with 0 or 1 depending on the predicate
// - The second pass runs an inclusive scan on the output data of the first pass using the CUB library
// - The third pass runs a kernel that copies the input data to the output data based on the indexes
// and output sizefrom the second pass
// TODO - we need to fuse the 3 kernels, calling CUB directly from the device code
template <typename Predicate>
std::vector<int> compact_stream_gpu(int* input_data, size_t size, Predicate predicate, int block_size) {
  NVTXScopedRange fn("compact_stream_gpu");
  Timer timer("compact_stream_gpu");
  CudaStream streamWrapper;
  cudaStream_t stream = streamWrapper.stream;
  auto input_data_indicators = make_cuda_unique<int>(size, true);

  create_input_data_indicators(streamWrapper, input_data, input_data_indicators.get(), size, block_size, predicate);

  cub_inclusive_scan(streamWrapper, input_data_indicators, input_data_indicators, size);
  auto output_size = input_data_indicators.get()[size - 1];
  std::vector<int> output_data(output_size);
  auto d_output_data = make_cuda_unique<int>(output_size, true);

  create_output_data(streamWrapper, input_data, input_data_indicators.get(), d_output_data.get(), size, block_size,
                     predicate);

  cudaCheck(cudaMemcpyAsync(output_data.data(), d_output_data.get(), output_size * sizeof(int), cudaMemcpyDeviceToHost,
                            stream));
  cudaCheck(cudaStreamSynchronize(stream));
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  return output_data;
}

int main(int argc, char* argv[]) {
  NVTXScopedRange fn("main");
  Args args;
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  if (parse_args(argc, argv, args, deviceProp) != 0) {
    return 1;
  }
  auto predicate = [](int x) -> bool { return x % 2 == 0; };
  auto input_data = generate_input_data_cuda(args.size);
  auto output_data_cpu_serial = compact_stream_cpu_serial(input_data.get(), args.size, predicate);

  if (args.debug_print) {
    print_vector("Input data", input_data.get(), args.size);
    print_vector("Output data CPU serial", output_data_cpu_serial);
  }

  auto predicate_gpu = [] __device__(int x) { return x % 2 == 0; };
  auto output_data_gpu = compact_stream_gpu(input_data.get(), args.size, predicate_gpu, args.block_size);
  if (args.debug_print) {
    print_vector("Output data GPU", output_data_gpu);
  }

  bool result = verify_result(output_data_cpu_serial.data(), output_data_gpu.data(), output_data_cpu_serial.size());
  printf("GPU results match: %s\n", result ? "true" : "false");

  return result ? 0 : 1;
}
