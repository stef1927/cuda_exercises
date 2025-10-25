#include <nvtx3/nvToolsExt.h>
#include <omp.h>

#include <cassert>
#include <cstdlib>
#include <execution>
#include <functional>
#include <ranges>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.h"
#include "stream_compaction_utils.h"


struct Args {
  int size;
  int omp_chunk_size;  // OpenMP chunk size
  bool debug_print;
};


int parse_args(int argc, char* argv[], Args& args) {
  argparse::ArgumentParser program("stream_compaction_cpu");
  program.add_argument("--size")
      .help("The size of the array to scan")
      .scan<'i', int>()
      .default_value(1 << 24)
      .store_into(args.size);
  program.add_argument("--omp-chunk-size")
      .help("The chunk size")
      .scan<'i', int>()
      .default_value(1024 * 1024)
      .store_into(args.omp_chunk_size);
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
  printf("  OMP chunk size: %d\n", args.omp_chunk_size);
  return 0;
}


// This runs in parallel on the CPU using the STL, which by default uses OpenMP as well, but we could make it run
// on the GPU too by installing the HPC SDK and compiling with nvc++ and stdpar=gpu: nvc++ -std=c++20 -stdpar=gpu -O3
std::vector<int> compact_stream_cpu_parallel_stl(int* input_data, size_t size, std::function<bool(int)> predicate) {
  nvtxRangePushA("compact_stream_cpu_parallel_stl");
  Timer timer("compact_stream_cpu_parallel_stl");
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
                [predicate, input_data, output_data_indicators, &output_data](size_t i) {
                  if (predicate(input_data[i])) {
                    int index = output_data_indicators[i];
                    output_data[index - 1] = input_data[i];
                  }
                });
  nvtxRangePop();
  return output_data;
}


// This runs in parallel on the CPU using OpenMP.
// TODO - fuse the 3 blocks together and optimize it further.
std::vector<int> compact_stream_cpu_parallel_omp(int* input_data, size_t size, std::function<bool(int)> predicate,
                                                 int chunk_size) {
  nvtxRangePushA("compact_stream_cpu_parallel_omp");
  Timer timer("compact_stream_cpu_parallel_omp");
  std::vector<int> output_data_indicators(size);
  std::vector<int> output_data;

  nvtxRangePushA("create_indicators");
#pragma omp parallel for schedule(static, chunk_size) default(shared)
  for (int i = 0; i < size; i++) {
    output_data_indicators[i] = predicate(input_data[i]) ? 1 : 0;
  }
  nvtxRangePop();

  nvtxRangePushA("inclusive_scan");
  std::inclusive_scan(std::execution::par, output_data_indicators.begin(), output_data_indicators.end(),
                      output_data_indicators.begin());
  nvtxRangePop();

  auto output_size = output_data_indicators.back();
  output_data.resize(output_size);

  nvtxRangePushA("compact_output");
#pragma omp parallel for schedule(static, chunk_size) default(shared)
  for (int i = 0; i < size; i++) {
    if (predicate(input_data[i])) {
      output_data[output_data_indicators[i] - 1] = input_data[i];
    }
  }
  nvtxRangePop();
  nvtxRangePop();  // For compact_stream_cpu_parallel_omp
  return output_data;
}


int main(int argc, char* argv[]) {
  nvtxRangePushA("main");
  Args args;
  if (parse_args(argc, argv, args) != 0) {
    nvtxRangePop();
    return 1;
  }
  auto predicate = [](int x) -> bool { return x % 2 == 0; };
  auto input_data = generate_input_data(args.size);
  auto output_data_cpu_serial = compact_stream_cpu_serial(input_data.data(), args.size, predicate);
  auto output_data_cpu_parallel_stl = compact_stream_cpu_parallel_stl(input_data.data(), args.size, predicate);
  auto output_data_cpu_parallel_omp =
      compact_stream_cpu_parallel_omp(input_data.data(), args.size, predicate, args.omp_chunk_size);

  if (args.debug_print) {
    print_vector("Input data", input_data.data(), args.size);
    print_vector("Output data CPU serial", output_data_cpu_serial);
    print_vector("Output data CPU parallel STL", output_data_cpu_parallel_stl);
    print_vector("Output data CPU parallel OMP", output_data_cpu_parallel_omp);
  }

  bool result =
      verify_result(output_data_cpu_serial.data(), output_data_cpu_parallel_stl.data(), output_data_cpu_serial.size());
  printf("CPU serial and parallel STL results match: %s\n", result ? "true" : "false");
  result =
      verify_result(output_data_cpu_serial.data(), output_data_cpu_parallel_omp.data(), output_data_cpu_serial.size());
  printf("CPU serial and parallel OMP results match: %s\n", result ? "true" : "false");

  nvtxRangePop();
  return result ? 0 : 1;
}
