#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
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

namespace cg = cooperative_groups;

const int WARP_SIZE = 32;
enum class KernelType { CUB, CTA_FUNCTIONS };


std::string kernel_type_to_string(KernelType kernel_type) {
  static const std::map<KernelType, std::string> _kernel_type_to_string = {
      {KernelType::CUB, "CUB library"},
      {KernelType::CTA_FUNCTIONS, "CTA collective functions"},
  };

  return _kernel_type_to_string.at(kernel_type);
}


KernelType string_to_kernel_type(const std::string& kernel_type) {
  static const std::map<std::string, KernelType> _string_to_kernel_type = {
      {"cub", KernelType::CUB},
      {"cta", KernelType::CTA_FUNCTIONS},
  };

  std::string lower_kernel_type = kernel_type;
  std::transform(lower_kernel_type.begin(), lower_kernel_type.end(), lower_kernel_type.begin(), ::tolower);

  return _string_to_kernel_type.at(lower_kernel_type);
}

struct Args {
  int size;
  int block_size;
  KernelType kernel_type;
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
  program.add_argument("--kernel-type")
      .help("The type of GPU kernel to use: cub or cta")
      .choices("cub", "cta")
      .default_value("cta")
      .store_into(kernel_type);
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

  args.kernel_type = string_to_kernel_type(kernel_type);

  if (args.kernel_type == KernelType::CTA_FUNCTIONS) {
    if (deviceProp.major < 5) {
      throw std::runtime_error("CTA collective functions are only supported on CC 5.0 and above");
    }
    if (args.block_size % WARP_SIZE != 0) {
      throw std::runtime_error("Block size must be divisible by the warp size for CTA collective functions kernel");
    }
    // Because we use warps to scan the warp sums, we cannot have more than WARP_SIZE warps in a
    // block
    if (args.block_size / WARP_SIZE > WARP_SIZE) {
      throw std::runtime_error("Block size must be less than or equal to " + std::to_string(WARP_SIZE * WARP_SIZE) +
                               " for CTA collective functions kernel");
    }
  }

  printf("Arguments:\n");
  printf("  Size: %d\n", args.size);
  printf("  Block size: %d\n", args.block_size);
  printf("  Kernel type: %s\n", kernel_type.c_str());
  return 0;
}


std::vector<int> generate_input_data(int size) {
  std::default_random_engine generator(786);
  std::uniform_int_distribution<int> distribution(0, 100);
  std::vector<int> input_data(size);
  for (int i = 0; i < size; i++) {
    input_data[i] = distribution(generator);
  }
  return input_data;
}


bool verify_result(std::vector<int>& output_data_cpu, std::vector<int>& output_data_gpu) {
  if (output_data_cpu.size() != output_data_gpu.size()) {
    printf("Output data size mismatch: %zu vs %zu\n", output_data_cpu.size(), output_data_gpu.size());
    return false;
  }
  for (size_t i = 0; i < output_data_cpu.size(); i++) {
    if (output_data_cpu[i] != output_data_gpu[i]) {
      printf("Output data mismatch at index %zu: %d vs %d, diff: %d\n", i, output_data_cpu[i], output_data_gpu[i],
             output_data_cpu[i] - output_data_gpu[i]);
      return false;
    }
  }
  return true;
}


std::vector<int> cpu_inclusive_scan(const std::vector<int>& input_data) {
  Timer timer("inclusive_scan on the CPU");
  std::vector<int> output_data(input_data.size());
  std::inclusive_scan(input_data.begin(), input_data.end(), output_data.begin());
  return output_data;
}


// A kernel that demonstrates the use of the CTA built-in functions to perform an inclusive scan:
// Each warp will call inclusive_scan() as documented in the CUDA C++ programming guide:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inclusive-scan-and-exclusive-scan
// This will assign a value to each thread which is correct within the warp. The value of the last thread
// of each warp will contain the warp sum, which is saved in shared memory. At this point, the first warp
// will scan the warp sums from shared memory and update shared memory so that shared memory will contain
// a scan of all the warp sums, which are added to the value of each thread except those in the first warp
// and used to update the output array. In addition, the sum of the block (the last warp sum or the value
// of the last thread) will be passed into the block sums array if it is not null. This will be used by
// the caller to scan the block sums, refer to the calling function for more details.
__global__ void cta_functions_inclusive_scan_kernel(int* d_input_data, int* d_output_data, int size,
                                                    int* d_block_sums = nullptr) {
  extern __shared__ unsigned int sdata[];  // partial sums of each warp, one per warp in the block
  auto thread_block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(thread_block);
  auto tid = thread_block.group_index().x * thread_block.group_dim().x + thread_block.thread_index().x;
  auto num_warps = thread_block.group_dim().x / WARP_SIZE;
  auto warp_id = thread_block.thread_index().x / WARP_SIZE;
  auto lane_id = thread_block.thread_index().x % WARP_SIZE;

  // First scan each warp tile using the CTA built-int function
  int input_val = (tid < size) ? d_input_data[tid] : 0;
  unsigned int output_val = cg::inclusive_scan(warp, input_val);

  // Make the last thread of each warp write the warp sum to shared memory
  if (lane_id == WARP_SIZE - 1) {
    sdata[warp_id] = output_val;
  }

  thread_block.sync();

  // Use the same built-in function to scan the warp sums using the threads of the first warp
  // up to the number of warps in the block
  if (warp_id == 0) {
    unsigned int warp_sum = (lane_id < num_warps) ? sdata[lane_id] : 0;
    unsigned int scanned_sum = cg::inclusive_scan(warp, warp_sum);
    if (lane_id < num_warps) {
      sdata[lane_id] = scanned_sum;
    }
  }

  thread_block.sync();

  // Now that shared memory contains the increments of each warp, perform a uniform add across the warps:
  // except for the first one, add the sum of the previous warp to all the output values and update the output array
  int block_sum = 0;
  if (warp_id > 0) {
    block_sum = sdata[warp_id - 1];
  }

  output_val += block_sum;

  if (tid < size)
    d_output_data[tid] = output_val;

  // If a block sums device array is passed in, then save the block sum to the array, this is either the sum
  // of the last warp from shared memory or the value of the last thread in the block
  if (d_block_sums != nullptr && thread_block.thread_index().x == thread_block.group_dim().x - 1) {
    d_block_sums[thread_block.group_index().x] = output_val;
  }
}


// This is a simple kernel that adds the block sum to all values in a block, refer to the calling function
// for more details. Here d_output_data points to the first block (index 1), whilst d_block_sums contains
// the sums of the blocks starting at block with index zero.
__global__ void uniform_add_kernel(int* d_output_data, int size, int* d_block_sums) {
  __shared__ int block_sum;
  auto thread_block = cg::this_thread_block();
  auto tid = thread_block.group_index().x * thread_block.group_dim().x + thread_block.thread_index().x;

  // only the first thread loads the value from device memory since all threads use the same value
  if (thread_block.thread_index().x == 0) {
    block_sum = d_block_sums[thread_block.group_index().x];
  }
  thread_block.sync();

  if (tid < size) {
    d_output_data[tid] += block_sum;
  }
}

// This function demonstrates the use of the CTA built-in functions to perform an inclusive scan.
// First, each block is scanned individually, returning the sum of each block as well. Afterwards,
// the block sums are also scanned so that they can be added to the output data.
void cta_functions_inclusive_scan(CudaStream& streamWrapper, CudaUniquePtr<int>& d_input_data,
                                  CudaUniquePtr<int>& d_output_data, int size, int block_size, bool debug_print) {
  cudaStream_t stream = streamWrapper.stream;
  CudaEventRecorder recorder = streamWrapper.record("inclusive_scan on the GPU using the CTA intrisics");
  dim3 dimBlock(block_size);
  dim3 dimGrid((size + block_size - 1) / block_size);
  int numWarps = dimBlock.x / WARP_SIZE;
  int shared_mem_size = numWarps * sizeof(unsigned int);  // for the sums of each warp
  auto d_block_sums = make_cuda_unique<int>(dimGrid.x);

  assert(dimBlock.x % WARP_SIZE == 0);
  assert(numWarps <= WARP_SIZE);

  if (debug_print) {
    printf("Running CTA functions inclusive scan kernel 1 with %d blocks and %d threads per block\n", dimGrid.x,
           dimBlock.x);
  }

  // Reduce each block individually and retrieve the block sums
  cta_functions_inclusive_scan_kernel<<<dimGrid, dimBlock, shared_mem_size, stream>>>(
      d_input_data.get(), d_output_data.get(), size, d_block_sums.get());

  // If there is only one block, there is nothing else to do because we don't need to add the sums of any previous
  // block.
  if (dimGrid.x <= 1) {
    return;
  }

  // Reduce the block sums using the same kernel if it fits in a single block, otherwise calls this function recursively
  if (dimGrid.x <= dimBlock.x) {
    dim3 dimGridSums((dimGrid.x + dimBlock.x - 1) / dimBlock.x);
    if (debug_print) {
      printf("Running CTA functions inclusive scan kernel 2 with %d blocks and %d threads per block\n", dimGridSums.x,
             dimBlock.x);
    }
    cta_functions_inclusive_scan_kernel<<<dimGridSums, dimBlock, shared_mem_size, stream>>>(
        d_block_sums.get(), d_block_sums.get(), dimGrid.x, nullptr);

    if (debug_print) {
      std::vector<int> h_block_sums(dimGrid.x);
      cudaCheck(cudaMemcpyAsync(h_block_sums.data(), d_block_sums.get(), dimGrid.x * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaStreamSynchronize(stream));
      print_vector("Block sums", h_block_sums);
    }
  } else {
    if (debug_print) {
      printf("Recursive call into cta_functions_inclusive_scan with %d blocks\n", dimGrid.x);
    }
    cta_functions_inclusive_scan(streamWrapper, d_block_sums, d_block_sums, dimGrid.x, block_size, debug_print);
  }


  // Add the block sums to the output data, skip the first block because the values of the first block are already
  // correct.
  dim3 dimGridAdd(dimGrid.x - 1);
  if (debug_print) {
    printf("Running uniform add kernel with %d blocks and %d threads per block\n", dimGridAdd.x, dimBlock.x);
  }
  uniform_add_kernel<<<dimGridAdd, dimBlock, 0, stream>>>(d_output_data.get() + block_size, size - block_size,
                                                          d_block_sums.get());
}


// Use the CUB library inclusive sum function
void cub_inclusive_scan(CudaStream& streamWrapper, CudaUniquePtr<int>& d_input_data, CudaUniquePtr<int>& d_output_data,
                        int size) {
  cudaStream_t stream = streamWrapper.stream;
  CudaEventRecorder recorder = streamWrapper.record("inclusive_scan on the GPU using the CUB library");

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cudaCheck(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_input_data.get(), d_output_data.get(), size,
                                          stream));

  // Allocate temporary storage
  auto d_temp_storage = make_cuda_unique<char>(temp_storage_bytes);

  // Run exclusive prefix sum
  cudaCheck(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_input_data.get(),
                                          d_output_data.get(), size, stream));
}


std::vector<int> gpu_inclusive_scan(const std::vector<int>& input_data, const Args& args) {
  Timer timer("inclusive_scan on the GPU using " + kernel_type_to_string(args.kernel_type));
  std::vector<int> output_data_gpu(input_data.size());
  CudaStream streamWrapper;
  cudaStream_t stream = streamWrapper.stream;

  auto d_input_data = make_cuda_unique<int>(input_data.size());
  auto d_output_data = make_cuda_unique<int>(input_data.size());

  // Copy input data to device
  cudaCheck(cudaMemcpyAsync(d_input_data.get(), input_data.data(), input_data.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream));

  {
    switch (args.kernel_type) {
      case KernelType::CUB: {
        cub_inclusive_scan(streamWrapper, d_input_data, d_output_data, input_data.size());
        break;
      }
      case KernelType::CTA_FUNCTIONS: {
        cta_functions_inclusive_scan(streamWrapper, d_input_data, d_output_data, input_data.size(), args.block_size,
                                     args.debug_print);
        break;
      }
    }
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

  // Run inclusive scan on GPU
  std::vector<int> output_data_gpu = gpu_inclusive_scan(input_data, args);

  if (args.debug_print) {
    print_vector("Input", input_data);
    print_vector("Output CPU", output_data_cpu);
    print_vector("Output GPU", output_data_gpu);
  }

  // Verify results
  bool result = verify_result(output_data_cpu, output_data_gpu);
  printf("Results match: %s\n", result ? "true" : "false");

  return result ? 0 : 1;
}
