#ifndef STREAM_COMPACTION_UTILS_H
#define STREAM_COMPACTION_UTILS_H

#include <functional>
#include <vector>

// Generate input data for stream compaction
std::vector<int> generate_input_data(int size);

// Verify that two result arrays match
bool verify_result(int* output_data_cpu, int* output_data_gpu, size_t size);

// CPU serial implementation of stream compaction
std::vector<int> compact_stream_cpu_serial(int* input_data, size_t size, std::function<bool(int)> predicate);

#endif  // STREAM_COMPACTION_UTILS_H
