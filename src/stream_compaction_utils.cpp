

#include "stream_compaction_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "cpp_utils.hpp"

std::vector<int> generate_input_data(int size) {
  std::vector<int> input_data(size);
  std::default_random_engine generator(786);
  std::uniform_int_distribution<int> distribution(0, 100);
  for (int i = 0; i < size; i++) {
    input_data[i] = distribution(generator);
  }
  return input_data;
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

// This runs serially on the CPU
std::vector<int> compact_stream_cpu_serial(int* input_data, size_t size, std::function<bool(int)> predicate) {
  Timer timer("compact_stream_cpu_serial");
  std::vector<int> output_data;
  std::copy_if(input_data, input_data + size, std::back_inserter(output_data), predicate);
  return output_data;
}
