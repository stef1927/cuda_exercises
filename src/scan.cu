#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cub/cub.cuh>  // or equivalently <cub/device/device_scan.cuh>
#include <random>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.h"
#include "cuda_utils.h"

struct Args {
  int size;
  int block_size;
};

int parse_args(int argc, char* argv[], Args& args, cudaDeviceProp& deviceProp) {
  argparse::ArgumentParser program("scan");
  program.add_argument("--size")
      .help("size of the array to scan")
      .scan<'i', int>()
      .default_value(1 << 24)
      .store_into(args.size);
  program.add_argument("--block-size")
      .help("block size")
      .scan<'i', int>()
      .default_value(256)
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

void print_vector(const char* name, const std::vector<int>& output_data) {
  printf("%s: [", name);
  for (int i = 0; i < output_data.size(); i++) {
    printf("%d ", output_data[i]);
  }
  printf("\n");
}

std::vector<int> generate_input_data(int size) {
  std::default_random_engine generator(786);
  std::uniform_int_distribution<int> distribution(0, 100);
  std::vector<int> input_data(size);
  for (int i = 0; i < size; i++) {
    input_data[i] = distribution(generator);
  }
  // print_vector("Input", input_data);
  return input_data;
}

bool verify_result(std::vector<int>& output_data_cpu, std::vector<int>& output_data_gpu) {
  // print_vector("CPU", output_data_cpu);
  // print_vector("GPU", output_data_gpu);
  return std::equal(output_data_cpu.begin(), output_data_cpu.end(), output_data_gpu.begin());
}

std::vector<int> cpu_inclusive_scan(const std::vector<int>& input_data) {
  Timer timer("inclusive_scan on the CPU");
  std::vector<int> output_data(input_data.size());
  std::inclusive_scan(input_data.begin(), input_data.end(), output_data.begin());
  return output_data;
}

std::vector<int> cub_inclusive_scan(const std::vector<int>& input_data) {
  Timer timer("inclusive_scan on the GPU using the CUB library");
  std::vector<int> output_data_gpu(input_data.size());
  CudaStream streamWrapper;
  cudaStream_t stream = streamWrapper.stream;

  auto d_input_data = make_cuda_unique<int>(input_data.size());
  auto d_output_data = make_cuda_unique<int>(input_data.size());

  // Copy input data to device
  cudaCheck(cudaMemcpyAsync(d_input_data.get(), input_data.data(), input_data.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream));

  {
    CudaEventRecorder recorder = streamWrapper.record("inclusive_scan on the GPU using the CUB library");

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cudaCheck(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_input_data.get(), d_output_data.get(),
                                            input_data.size(), stream));

    // Allocate temporary storage
    auto d_temp_storage = make_cuda_unique<char>(temp_storage_bytes);

    // Run exclusive prefix sum
    cudaCheck(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_input_data.get(),
                                            d_output_data.get(), input_data.size(), stream));
  }

  cudaCheck(cudaMemcpyAsync(output_data_gpu.data(), d_output_data.get(), input_data.size() * sizeof(int),
                            cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaStreamSynchronize(stream));

  return output_data_gpu;
}

int main(int argc, char* argv[]) {
  Args args;
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  if (parse_args(argc, argv, args, deviceProp) != 0) {
    return 1;
  }

  // Run inclusive scan on CPU
  const std::vector<int> input_data = generate_input_data(args.size);
  std::vector<int> output_data_cpu = cpu_inclusive_scan(input_data);

  // Run inclusive scan on GPU using the CUB library
  std::vector<int> output_data_gpu = cub_inclusive_scan(input_data);

  // Verify results
  bool result = verify_result(output_data_cpu, output_data_gpu);
  printf("Results match: %s\n", result ? "true" : "false");

  return result ? 0 : 1;
}