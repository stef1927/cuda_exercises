#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cmath>
#include <random>
#include <ranges>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.hpp"
#include "cuda_utils.cuh"

struct Args {
  int N;
  int block_size;
};

using CartesianCoordFloat = float2;
using CartesianCoordHalf = __half2;
using PolarCoordFloat = float2;  // r in .x, theta in .y
using PolarCoordHalf = __half2;  // r in .x, theta in .y

enum class KernelType {
  NAIVE,
  OPTIMIZED,
};

int parse_args(int argc, char* argv[], Args& args) {
  argparse::ArgumentParser program("coordinates");
  program.add_argument("--N")
      .help("The number of points")
      .scan<'i', int>()
      .default_value(1024 << 12)
      .store_into(args.N);
  program.add_argument("--block-size")
      .help("The block size")
      .scan<'i', int>()
      .default_value(128)
      .store_into(args.block_size);
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  printf("Arguments:\n");
  printf("  N: %d\n", args.N);
  printf("  Block size: %d\n", args.block_size);
  return 0;
}


std::vector<CartesianCoordFloat> generate_cartesian_coordinates(int N) {
  Timer timer("generate_cartesian_coordinates");
  std::vector<CartesianCoordFloat> cartesian_coordinates;
  cartesian_coordinates.reserve(N);
  std::default_random_engine generator(786);
  std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
  for (int i = 0; i < N; i++) {
    float x = distribution(generator);
    float y = distribution(generator);
    cartesian_coordinates.push_back(make_float2(x, y));
  }
  return cartesian_coordinates;
}

std::vector<PolarCoordFloat> cartesian_to_polar_cpu(const std::vector<CartesianCoordFloat>& cartesian_coordinates) {
  Timer timer("cartesian_to_polar_cpu");
  std::vector<PolarCoordFloat> polar_coordinates(cartesian_coordinates.size(), make_float2(0.0f, 0.0f));
  std::transform(cartesian_coordinates.begin(), cartesian_coordinates.end(), polar_coordinates.begin(),
                 [](const CartesianCoordFloat& cartesian_coordinate) -> PolarCoordFloat {
                   float r = sqrt(cartesian_coordinate.x * cartesian_coordinate.x +
                                  cartesian_coordinate.y * cartesian_coordinate.y);
                   float theta = atan2(cartesian_coordinate.y, cartesian_coordinate.x);
                   return make_float2(r, theta);
                 });
  return polar_coordinates;
}


bool verify_results(const std::vector<PolarCoordFloat>& polar_coordinates,
                    const std::vector<PolarCoordFloat>& polar_coordinates_ref, double tolerance = 1e-4) {
  auto ret =
      std::ranges::equal(polar_coordinates, polar_coordinates_ref,
                         [tolerance](const PolarCoordFloat& a, const PolarCoordFloat& b) -> bool {
                           auto ret = std::fabs(a.x - b.x) < tolerance && std::fabs(a.y - b.y) < tolerance;
                           if (!ret) {
                             printf("Divergence! r: %5.4f != %5.4f (Diff %5.4f), theta: %5.4f != %5.4f (Diff %5.4f)",
                                    b.x, a.x, std::fabs(a.x - b.x), b.y, a.y, std::fabs(a.y - b.y));
                           }
                           return ret;
                         });
  printf("Results verified: %s\n", ret ? "true" : "false");
  return ret;
}


__global__ void cartesian_to_polar_kernel_naive(CartesianCoordFloat* _cartesian_coordinates,
                                                PolarCoordFloat* d_polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoordFloat coord = _cartesian_coordinates[idx];
    float r = sqrt(coord.x * coord.x + coord.y * coord.y);
    float theta = atan2(coord.y, coord.x);
    PolarCoordFloat polar_coord = make_float2(r, theta);
    d_polar_coordinates[idx] = polar_coord;
  }
}


__global__ void cartesian_to_polar_kernel_optimized_full(CartesianCoordFloat* __restrict__ _cartesian_coordinates,
                                                         PolarCoordFloat* __restrict__ _polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoordFloat coord = _cartesian_coordinates[idx];
    float r = hypotf(coord.x, coord.y);
    float theta = atan2f(coord.y, coord.x);
    PolarCoordFloat polar_coord = make_float2(r, theta);
    _polar_coordinates[idx] = polar_coord;
  }
}

void run_kernel_full_precision(std::vector<CartesianCoordFloat>& cartesian_coordinates,
                               std::vector<PolarCoordFloat>& polar_coordinates, int block_size, KernelType kernelType) {
  Timer timer("cartesian_to_polar_kernel");
  auto N = cartesian_coordinates.size();
  auto streamWrapper = CudaStream();
  cudaStream_t stream = streamWrapper.stream;
  auto d_cartesian_coordinates = make_cuda_unique<CartesianCoordFloat>(N);
  cudaCheck(cudaMemcpyAsync(d_cartesian_coordinates.get(), cartesian_coordinates.data(),
                            N * sizeof(CartesianCoordFloat), cudaMemcpyHostToDevice, streamWrapper.stream));
  auto d_polar_coordinates = make_cuda_unique<PolarCoordFloat>(N);


  auto recorder = streamWrapper.record("cartesian_to_polar_kernel");
  dim3 dimBlock(block_size);
  dim3 dimGrid((N + block_size - 1) / block_size);

  switch (kernelType) {
    case KernelType::NAIVE: {
      cartesian_to_polar_kernel_naive<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(d_cartesian_coordinates.get(),
                                                                                      d_polar_coordinates.get(), N);
      break;
    }
    case KernelType::OPTIMIZED: {
      cartesian_to_polar_kernel_optimized_full<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(
          d_cartesian_coordinates.get(), d_polar_coordinates.get(), N);
      break;
    }
    default: {
      throw std::runtime_error("Invalid kernel type");
    }
  }
  cudaCheck(cudaGetLastError());
  recorder.close();

  cudaCheck(cudaMemcpyAsync(polar_coordinates.data(), d_polar_coordinates.get(), N * sizeof(PolarCoordFloat),
                            cudaMemcpyDeviceToHost, streamWrapper.stream));
  cudaCheck(cudaStreamSynchronize(streamWrapper.stream));
}

__global__ void cartesian_to_polar_kernel_optimized_half(CartesianCoordHalf* __restrict__ _cartesian_coordinates,
                                                         PolarCoordHalf* __restrict__ _polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoordHalf coord = _cartesian_coordinates[idx];
    __half r = hypotf(coord.x, coord.y);
    __half theta = atan2f(coord.y, coord.x);
    PolarCoordHalf polar_coord = make_half2(r, theta);
    _polar_coordinates[idx] = polar_coord;
  }
}


void run_kernel_half_precision(std::vector<CartesianCoordFloat>& cartesian_coordinates,
                               std::vector<PolarCoordFloat>& polar_coordinates, int block_size) {
  Timer timer("cartesian_to_polar_kernel");
  auto N = cartesian_coordinates.size();
  auto streamWrapper = CudaStream();
  cudaStream_t stream = streamWrapper.stream;
  auto cartesian_coordinates_half = std::vector<CartesianCoordHalf>(N, make_half2(0.0f, 0.0f));
  std::transform(cartesian_coordinates.begin(), cartesian_coordinates.end(), cartesian_coordinates_half.begin(),
                 [](const CartesianCoordFloat& cartesian_coordinate) -> CartesianCoordHalf {
                   return make_half2(__float2half(cartesian_coordinate.x), __float2half(cartesian_coordinate.y));
                 });
  auto d_cartesian_coordinates = make_cuda_unique<CartesianCoordHalf>(N);
  cudaCheck(cudaMemcpyAsync(d_cartesian_coordinates.get(), cartesian_coordinates.data(), N * sizeof(CartesianCoordHalf),
                            cudaMemcpyHostToDevice, streamWrapper.stream));
  auto d_polar_coordinates = make_cuda_unique<PolarCoordHalf>(N);

  auto recorder = streamWrapper.record("cartesian_to_polar_kernel");
  dim3 dimBlock(block_size);
  dim3 dimGrid((N + block_size - 1) / block_size);

  cartesian_to_polar_kernel_optimized_half<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(
      d_cartesian_coordinates.get(), d_polar_coordinates.get(), N);

  cudaCheck(cudaGetLastError());
  recorder.close();

  auto polar_coordinates_half = std::vector<PolarCoordHalf>(N, make_half2(0.0f, 0.0f));
  cudaCheck(cudaMemcpyAsync(polar_coordinates_half.data(), d_polar_coordinates.get(), N * sizeof(PolarCoordHalf),
                            cudaMemcpyDeviceToHost, streamWrapper.stream));
  cudaCheck(cudaStreamSynchronize(streamWrapper.stream));
  std::transform(polar_coordinates_half.begin(), polar_coordinates_half.end(), polar_coordinates.begin(),
                 [](const PolarCoordHalf& polar_coordinate) -> PolarCoordFloat {
                   return make_float2(__half2float(polar_coordinate.x), __half2float(polar_coordinate.y));
                 });
}

int main(int argc, char* argv[]) {
  int ret = 0;
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  Args args;
  if (parse_args(argc, argv, args) != 0) {
    return 1;
  }

  auto cartesian_coordinates = generate_cartesian_coordinates(args.N);
  auto polar_coordinates_ref = cartesian_to_polar_cpu(cartesian_coordinates);
  auto polar_coordinates = std::vector<PolarCoordFloat>(args.N, make_float2(0.0f, 0.0f));

  run_kernel_full_precision(cartesian_coordinates, polar_coordinates, args.block_size, KernelType::NAIVE);
  ret |= verify_results(polar_coordinates, polar_coordinates_ref);

  run_kernel_full_precision(cartesian_coordinates, polar_coordinates, args.block_size, KernelType::OPTIMIZED);
  ret |= verify_results(polar_coordinates, polar_coordinates_ref);

  run_kernel_half_precision(cartesian_coordinates, polar_coordinates, args.block_size);
  ret |= verify_results(polar_coordinates, polar_coordinates_ref);

  return ret;
}