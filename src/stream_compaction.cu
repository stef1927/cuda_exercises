
#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cub/cub.cuh>  // or equivalently <cub/device/device_scan.cuh>
#include <execution>
#include <functional>
#include <random>
#include <ranges>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.h"
#include "cuda_utils.h"


struct Args {
  int size;
  int block_size;
  bool debug_print;
};


int parse_args(int argc, char* argv[], Args& args, cudaDeviceProp& deviceProp) {
  argparse::ArgumentParser program("scan");
  std::string kernel_type;
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


CudaUniquePtr<int> generate_input_data(int size) {
  auto input_data = make_cuda_unique<int>(size, true);
  std::default_random_engine generator(786);
  std::uniform_int_distribution<int> distribution(0, 100);
  for (int i = 0; i < size; i++) {
    input_data.get()[i] = distribution(generator);
  }
  return input_data;
}

std::vector<int> compact_stream_cpu_serial(int* input_data, size_t size, std::function<bool(int)> predicate) {
  Timer timer("compact_stream_cpu_serial");
  std::vector<int> output_data;
  std::copy_if(input_data, input_data + size, std::back_inserter(output_data), predicate);
  return output_data;
}

std::vector<int> compact_stream_cpu_parallel(int* input_data, size_t size, std::function<bool(int)> predicate) {
  Timer timer("compact_stream_cpu_parallel");
  // Create a vector of 0 or 1 depending on predicate result
  std::vector<int> output_data_indicators(size);
  std::transform(std::execution::par, input_data, input_data + size, output_data_indicators.begin(),
                 [predicate](int x) { return predicate(x) ? 1 : 0; });

  // Run an inclusive scan, when the sum changes, that's the index of the next element to copy, the last index is the
  // size of the output data
  std::inclusive_scan(std::execution::par, output_data_indicators.begin(), output_data_indicators.end(),
                      output_data_indicators.begin());

  std::vector<int> output_data(output_data_indicators.back());
  auto indexes = std::views::iota((size_t)0, (size_t)size);
  std::for_each(std::execution::par, indexes.begin(), indexes.end(),
                [input_data, output_data_indicators, &output_data](size_t i) {
                  int prev_index = i > 0 ? output_data_indicators[i - 1] : 0;
                  int index = output_data_indicators[i];
                  if (index > prev_index) {
                    output_data[index - 1] = input_data[i];
                  }
                });
  return output_data;
}


bool verify_result(int* output_data_cpu, int* output_data_gpu, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (output_data_cpu[i] != output_data_gpu[i]) {
      printf("Output data mismatch at index %zu: %d vs %d, diff: %d\n", i, output_data_cpu[i], output_data_gpu[i],
             output_data_cpu[i] - output_data_gpu[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  Args args;
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  if (parse_args(argc, argv, args, deviceProp) != 0) {
    return 1;
  }
  auto input_data = generate_input_data(args.size);
  auto output_data_cpu_serial =
      compact_stream_cpu_serial(input_data.get(), args.size, [](int x) { return x % 2 == 0; });
  auto output_data_cpu_parallelizable =
      compact_stream_cpu_parallel(input_data.get(), args.size, [](int x) { return x % 2 == 0; });
  if (args.debug_print) {
    print_vector("Input data", input_data.get(), args.size);
    print_vector("Output data CPU serial", output_data_cpu_serial);
    print_vector("Output data CPU parallelizable", output_data_cpu_parallelizable);
  }

  bool result = verify_result(output_data_cpu_serial.data(), output_data_cpu_parallelizable.data(),
                              output_data_cpu_serial.size());
  printf("Results match: %s\n", result ? "true" : "false");
  return result ? 0 : 1;
  return 0;
}