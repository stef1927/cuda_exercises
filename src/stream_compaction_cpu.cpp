#include <omp.h>

#include <cassert>
#include <cstdlib>
#include <execution>
#include <functional>
#include <numeric>
#include <ranges>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.hpp"
#include "stream_compaction_utils.hpp"


struct Args {
  int size;
  int chunk_size;
  bool debug_print;
};


int parse_args(int argc, char* argv[], Args& args) {
  argparse::ArgumentParser program("stream_compaction_cpu");
  program.add_argument("--size")
      .help("The size of the array to scan")
      .scan<'i', int>()
      .default_value(1 << 24)
      .store_into(args.size);
  program.add_argument("--chunk-size")
      .help("The chunk size")
      .scan<'i', int>()
      .default_value(1024 * 1024)
      .store_into(args.chunk_size);
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
  printf("  Chunk size: %d\n", args.chunk_size);
  return 0;
}


// This runs in parallel on the CPU using the STL, which by default uses OpenMP as well, but we could make it run
// on the GPU too by installing the HPC SDK and compiling with nvc++ and stdpar=gpu: nvc++ -std=c++20 -stdpar=gpu -O3
std::vector<int> compact_stream_cpu_parallel_stl(const std::vector<int>& input_data,
                                                 std::function<bool(int)> predicate) {
  NVTXScopedRange fn("compact_stream_cpu_parallel_stl");
  Timer timer("compact_stream_cpu_parallel_stl");
  std::vector<int> input_data_prefix_sums(input_data.size());

  // Run an inclusive scan after applying the predicate.
  // When the sum changes, that's the index of the next element to copy, the last index is the
  // size of the output data
  std::transform_inclusive_scan(std::execution::par, input_data.begin(), input_data.end(),
                                input_data_prefix_sums.begin(), std::plus<int>(),
                                [predicate](int x) { return predicate(x) ? 1 : 0; });

  std::vector<int> output_data(input_data_prefix_sums.back());
  auto indexes = std::views::iota((size_t)0, input_data.size());
  std::for_each(std::execution::par, indexes.begin(), indexes.end(),
                [predicate, input_data, input_data_prefix_sums, &output_data](size_t i) {
                  if (predicate(input_data[i])) {
                    int index = input_data_prefix_sums[i];
                    output_data[index - 1] = input_data[i];
                  }
                });
  return output_data;
}


unsigned long long prefix_sum_with_pred_omp_chunk(const std::vector<int>& input_data, std::vector<int>& output_data,
                                                  std::function<bool(int)> predicate, int chunk_size) {
  NVTXScopedRange fn("prefix_sum_with_pred_omp_chunk");
  int tid = omp_get_thread_num();
  int num_threads = omp_get_num_threads();
  int start_index = tid * chunk_size;
  int end_index = (tid == num_threads - 1) ? input_data.size() : start_index + chunk_size;

  if (start_index >= input_data.size() || end_index > input_data.size()) {
    throw std::invalid_argument("Parameters out of range");
  }

  unsigned long long sum = 0;
  for (int i = start_index; i < end_index; i++) {
    if (predicate(input_data[i])) {
      ++sum;
    }
    output_data[i] = sum;
  }
  return sum;
}

void add_sum_of_previous_chunk(unsigned long long sum, std::vector<int>& output_data, int tid, int chunk_size) {
  NVTXScopedRange fn("add_sum_of_previous_chunk");
  int num_threads = omp_get_num_threads();
  int start_index = tid * chunk_size;
  int end_index = (tid == num_threads - 1) ? output_data.size() : start_index + chunk_size;

  if (start_index >= output_data.size() || end_index > output_data.size()) {
    throw std::invalid_argument("Parameters out of range");
  }

  for (int i = start_index; i < end_index; i++) {
    output_data[i] += sum;
  }
}


void prefix_sum_with_pred_omp(const std::vector<int>& input_data, std::vector<int>& output_data,
                              std::function<bool(int)> predicate) {
  NVTXScopedRange fn("prefix_sum_omp");

  if (input_data.size() == 0) {
    throw std::invalid_argument("Input data is empty");
  }

  // use the number of processors on the machine as the number of threads in the parallel regions
  int num_threads = omp_get_num_procs();
  int chunk_size = input_data.size() / num_threads;
  auto sums = std::vector<unsigned long long>(num_threads);

#pragma omp parallel default(shared) num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    sums[tid] = prefix_sum_with_pred_omp_chunk(input_data, output_data, predicate, chunk_size);
  }  // implicit barrier here

  {
    NVTXScopedRange fn("scan_sums");
    // scan the sum, sequential for now
    for (int i = 1; i < sums.size(); i++) {
      sums[i] += sums[i - 1];
    }
  }

#pragma omp parallel default(shared) num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    if (tid > 0) {
      add_sum_of_previous_chunk(sums[tid - 1], output_data, tid, chunk_size);
    }
  }  // implicit barrier here
}


std::vector<int> generate_output_data_omp(const std::vector<int>& input_data, std::vector<int>& output_data_prefix_sum,
                                          std::function<bool(int)> predicate) {
  NVTXScopedRange compact_output("generate_output_data_omp");
  std::vector<int> output_data(output_data_prefix_sum.back());
  int num_threads = omp_get_num_procs();
  int chunk_size = input_data.size() / num_threads;

#pragma omp parallel for schedule(static, chunk_size) default(shared)
  for (int i = 0; i < input_data.size(); i++) {
    if (predicate(input_data[i])) {
      output_data[output_data_prefix_sum[i] - 1] = input_data[i];
    }
  }
  return output_data;
}

std::vector<int> compact_stream_cpu_parallel_omp(const std::vector<int>& input_data,
                                                 std::function<bool(int)> predicate) {
  NVTXScopedRange fn("compact_stream_cpu_parallel_omp");
  Timer timer("compact_stream_cpu_parallel_omp");
  std::vector<int> output_data_prefix_sum(input_data.size());
  prefix_sum_with_pred_omp(input_data, output_data_prefix_sum, predicate);
  return generate_output_data_omp(input_data, output_data_prefix_sum, predicate);
}


int main(int argc, char* argv[]) {
  NVTXScopedRange fn("main");
  Args args;
  if (parse_args(argc, argv, args) != 0) {
    return 1;
  }
  auto predicate = [](int x) -> bool { return x % 2 == 0; };
  auto input_data = generate_input_data(args.size);
  auto output_data_cpu_serial = compact_stream_cpu_serial(input_data.data(), args.size, predicate);
  auto output_data_cpu_parallel_stl = compact_stream_cpu_parallel_stl(input_data, predicate);
  auto output_data_cpu_parallel_omp = compact_stream_cpu_parallel_omp(input_data, predicate);

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

  return result ? 0 : 1;
}
