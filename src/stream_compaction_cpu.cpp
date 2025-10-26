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
std::vector<int> compact_stream_stl(const std::vector<int>& input_data, std::function<bool(int)> predicate) {
  NVTXScopedRange fn("compact_stream_stl");
  Timer timer("compact_stream_stl");
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
                                                  std::function<bool(int)> predicate, int start_index, int end_index) {
  NVTXScopedRange fn("prefix_sum_with_pred_omp_chunk");

  unsigned long long sum = 0;
  for (int i = start_index; i < end_index; i++) {
    if (predicate(input_data[i])) {
      ++sum;
    }
    output_data[i] = sum;
  }
  return sum;
}

void add_sum_of_previous_chunk(unsigned long long sum, std::vector<int>& output_data, int start_index, int end_index) {
  NVTXScopedRange fn("add_sum_of_previous_chunk");

  for (int i = start_index; i < end_index; i++) {
    output_data[i] += sum;
  }
}


void generate_output_data_omp_chunk(const std::vector<int>& input_data, std::vector<int>& output_data,
                                    std::vector<int>& output_data_indexes, int start_index, int end_index) {
  NVTXScopedRange fn("generate_output_data_omp_chunk");

  int prev = start_index > 0 ? output_data_indexes[start_index - 1] : 0;
  for (int i = start_index; i < end_index; i++) {
    if (output_data_indexes[i] > prev) {
      output_data[output_data_indexes[i] - 1] = input_data[i];
      prev = output_data_indexes[i];
    }
  }
}

// This runs in parallel with all shared data
template <typename Predicate>
void compact_stream_omp_parallel(const std::vector<int>& input_data, std::vector<int>& output_data,
                                 std::vector<int>& output_data_indexes, std::vector<unsigned long long>& sums,
                                 Predicate predicate) {
  NVTXScopedRange fn("compact_stream_omp_parallel");

  int tid = omp_get_thread_num();
  int num_threads = omp_get_num_threads();
  int chunk_size = input_data.size() / num_threads;

  int start_index = tid * chunk_size;
  int end_index = (tid == num_threads - 1) ? input_data.size() : start_index + chunk_size;

  if (start_index >= input_data.size() || end_index > input_data.size()) {
    throw std::invalid_argument("Parameters out of range");
  }

  sums[tid] = prefix_sum_with_pred_omp_chunk(input_data, output_data_indexes, predicate, start_index, end_index);

#pragma omp barrier
#pragma omp master
  {
    NVTXScopedRange fn("scan_sums");
    // scan the sum, sequential for now
    for (int i = 1; i < sums.size(); i++) {
      sums[i] += sums[i - 1];
    }
  }
#pragma omp barrier


  if (tid > 0) {
    add_sum_of_previous_chunk(sums[tid - 1], output_data_indexes, start_index, end_index);
  }
#pragma omp barrier

#pragma omp master
  {
    output_data.resize(sums.back());
  }
#pragma omp barrier

  generate_output_data_omp_chunk(input_data, output_data, output_data_indexes, start_index, end_index);
}

template <typename Predicate>
std::vector<int> compact_stream_omp(const std::vector<int>& input_data, Predicate predicate) {
  NVTXScopedRange fn("compact_stream_omp");
  Timer timer("compact_stream_omp");
  int num_threads = omp_get_num_procs();
  std::vector<int> output_data;
  std::vector<int> output_data_indexes(input_data.size());
  std::vector<unsigned long long> sums(num_threads);
  output_data.reserve(input_data.size() / 2);

  if (input_data.size() == 0) {
    throw std::invalid_argument("Input data is empty");
  }

#pragma omp parallel default(shared) num_threads(num_threads)
  compact_stream_omp_parallel<Predicate>(input_data, output_data, output_data_indexes, sums, predicate);

  return output_data;
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
  auto output_data_cpu_parallel_stl = compact_stream_stl(input_data, predicate);
  auto output_data_cpu_parallel_omp = compact_stream_omp(input_data, predicate);

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
