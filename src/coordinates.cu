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

template <Numeric T>
struct PolarCoord {
  T r;      // radius
  T theta;  // angle in radians
};

template <Numeric T>
struct CartesianCoord {
  T x;
  T y;
};

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


std::vector<CartesianCoord<float>> generate_cartesian_coordinates(int N) {
  Timer timer("generate_cartesian_coordinates");
  std::vector<CartesianCoord<float>> cartesian_coordinates;
  cartesian_coordinates.reserve(N);
  std::default_random_engine generator(786);
  std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
  for (int i = 0; i < N; i++) {
    float x = distribution(generator);
    float y = distribution(generator);
    cartesian_coordinates.push_back(CartesianCoord<float>{x, y});
  }
  return cartesian_coordinates;
}

std::vector<PolarCoord<float>> cartesian_to_polar_cpu(const std::vector<CartesianCoord<float>>& cartesian_coordinates) {
  Timer timer("cartesian_to_polar_cpu");
  std::vector<PolarCoord<float>> polar_coordinates(cartesian_coordinates.size(), PolarCoord<float>{0.0f, 0.0f});
  std::transform(cartesian_coordinates.begin(), cartesian_coordinates.end(), polar_coordinates.begin(),
                 [](const CartesianCoord<float>& cartesian_coordinate) -> PolarCoord<float> {
                   float r = sqrt(cartesian_coordinate.x * cartesian_coordinate.x +
                                  cartesian_coordinate.y * cartesian_coordinate.y);
                   float theta = atan2(cartesian_coordinate.y, cartesian_coordinate.x);
                   return PolarCoord{r, theta};
                 });
  return polar_coordinates;
}


bool verify_results(const std::vector<PolarCoord<float>>& polar_coordinates,
                    const std::vector<PolarCoord<float>>& polar_coordinates_ref, double tolerance = 1e-4) {
  auto ret =
      std::ranges::equal(polar_coordinates, polar_coordinates_ref,
                         [tolerance](const PolarCoord<float>& a, const PolarCoord<float>& b) -> bool {
                           auto ret = std::fabs(a.r - b.r) < tolerance && std::fabs(a.theta - b.theta) < tolerance;
                           if (!ret) {
                             printf("Divergence! r: %5.4f != %5.4f (Diff %5.4f), theta: %5.4f != %5.4f (Diff %5.4f)",
                                    b.r, a.r, std::fabs(a.r - b.r), b.theta, a.theta, std::fabs(a.theta - b.theta));
                           }
                           return ret;
                         });
  printf("Results verified: %s\n", ret ? "true" : "false");
  return ret;
}


__global__ void cartesian_to_polar_kernel_naive(CartesianCoord<float>* _cartesian_coordinates,
                                                PolarCoord<float>* d_polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoord<float> coord = _cartesian_coordinates[idx];
    float r = sqrt(coord.x * coord.x + coord.y * coord.y);
    float theta = atan2(coord.y, coord.x);
    PolarCoord<float> polar_coord = PolarCoord<float>{r, theta};
    d_polar_coordinates[idx] = polar_coord;
  }
}


__global__ void cartesian_to_polar_kernel_optimized_full(CartesianCoord<float>* __restrict__ _cartesian_coordinates,
                                                         PolarCoord<float>* __restrict__ _polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoord<float> coord = _cartesian_coordinates[idx];
    float r = hypotf(coord.x, coord.y);
    float theta = atan2f(coord.y, coord.x);
    PolarCoord<float> polar_coord = PolarCoord<float>{r, theta};
    _polar_coordinates[idx] = polar_coord;
  }
}

void run_kernel_full_precision(std::vector<CartesianCoord<float>>& cartesian_coordinates,
                               std::vector<PolarCoord<float>>& polar_coordinates, int block_size,
                               KernelType kernelType) {
  Timer timer("cartesian_to_polar_kernel");
  auto N = cartesian_coordinates.size();
  auto streamWrapper = CudaStream();
  cudaStream_t stream = streamWrapper.stream;
  auto d_cartesian_coordinates = make_cuda_unique<CartesianCoord<float>>(N);
  cudaCheck(cudaMemcpyAsync(d_cartesian_coordinates.get(), cartesian_coordinates.data(),
                            N * sizeof(CartesianCoord<float>), cudaMemcpyHostToDevice, streamWrapper.stream));
  auto d_polar_coordinates = make_cuda_unique<PolarCoord<float>>(N);


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

  cudaCheck(cudaMemcpyAsync(polar_coordinates.data(), d_polar_coordinates.get(), N * sizeof(PolarCoord<float>),
                            cudaMemcpyDeviceToHost, streamWrapper.stream));
  cudaCheck(cudaStreamSynchronize(streamWrapper.stream));
}

__global__ void cartesian_to_polar_kernel_optimized_half(CartesianCoord<__half>* __restrict__ _cartesian_coordinates,
                                                         PolarCoord<__half>* __restrict__ _polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoord<__half> coord = _cartesian_coordinates[idx];
    __half r = hypotf(coord.x, coord.y);
    __half theta = atan2f(coord.y, coord.x);
    PolarCoord<__half> polar_coord = PolarCoord<__half>{r, theta};
    _polar_coordinates[idx] = polar_coord;
  }
}


void run_kernel_half_precision(std::vector<CartesianCoord<float>>& cartesian_coordinates,
                               std::vector<PolarCoord<float>>& polar_coordinates, int block_size) {
  Timer timer("cartesian_to_polar_kernel");
  auto N = cartesian_coordinates.size();
  auto streamWrapper = CudaStream();
  cudaStream_t stream = streamWrapper.stream;
  auto cartesian_coordinates_half = std::vector<CartesianCoord<__half>>(N, CartesianCoord<__half>{0.0f, 0.0f});
  std::transform(cartesian_coordinates.begin(), cartesian_coordinates.end(), cartesian_coordinates_half.begin(),
                 [](const CartesianCoord<float>& cartesian_coordinate) -> CartesianCoord<__half> {
                   return CartesianCoord<__half>{__float2half(cartesian_coordinate.x),
                                                 __float2half(cartesian_coordinate.y)};
                 });
  auto d_cartesian_coordinates = make_cuda_unique<CartesianCoord<__half>>(N);
  cudaCheck(cudaMemcpyAsync(d_cartesian_coordinates.get(), cartesian_coordinates.data(),
                            N * sizeof(CartesianCoord<__half>), cudaMemcpyHostToDevice, streamWrapper.stream));
  auto d_polar_coordinates = make_cuda_unique<PolarCoord<__half>>(N);

  auto recorder = streamWrapper.record("cartesian_to_polar_kernel");
  dim3 dimBlock(block_size);
  dim3 dimGrid((N + block_size - 1) / block_size);

  cartesian_to_polar_kernel_optimized_half<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(
      d_cartesian_coordinates.get(), d_polar_coordinates.get(), N);

  cudaCheck(cudaGetLastError());
  recorder.close();

  auto polar_coordinates_half = std::vector<PolarCoord<__half>>(N, PolarCoord<__half>{0.0f, 0.0f});
  cudaCheck(cudaMemcpyAsync(polar_coordinates_half.data(), d_polar_coordinates.get(), N * sizeof(PolarCoord<__half>),
                            cudaMemcpyDeviceToHost, streamWrapper.stream));
  cudaCheck(cudaStreamSynchronize(streamWrapper.stream));
  std::transform(polar_coordinates_half.begin(), polar_coordinates_half.end(), polar_coordinates.begin(),
                 [](const PolarCoord<__half>& polar_coordinate) -> PolarCoord<float> {
                   return PolarCoord<float>{__half2float(polar_coordinate.r), __half2float(polar_coordinate.theta)};
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
  auto polar_coordinates = std::vector<PolarCoord<float>>(args.N, PolarCoord<float>{0.0f, 0.0f});

  run_kernel_full_precision(cartesian_coordinates, polar_coordinates, args.block_size, KernelType::NAIVE);
  ret |= verify_results(polar_coordinates, polar_coordinates_ref);

  run_kernel_full_precision(cartesian_coordinates, polar_coordinates, args.block_size, KernelType::OPTIMIZED);
  ret |= verify_results(polar_coordinates, polar_coordinates_ref);

  run_kernel_half_precision(cartesian_coordinates, polar_coordinates, args.block_size);
  ret |= verify_results(polar_coordinates, polar_coordinates_ref);

  return ret;
}