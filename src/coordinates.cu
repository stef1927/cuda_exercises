#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cmath>
#include <cuda/std/cmath>
#include <cuda/std/concepts>
#include <format>
#include <random>
#include <vector>

#include "argparse.hpp"
#include "cpp_utils.hpp"
#include "cuda_utils.cuh"

struct Args {
  int N;
  int block_size;
  std::string precision;
};

using CartesianCoordFloat = float2;
using CartesianCoordDouble = double2;
using CartesianCoordBf16 = __nv_bfloat162;

// r in .x, theta in .y
using PolarCoordFloat = float2;
using PolarCoordDouble = double2;
using PolarCoordBf16 = __nv_bfloat162;

template <typename T>
concept CartesianCoord = cuda::std::is_same_v<T, CartesianCoordFloat> ||
                         cuda::std::is_same_v<T, CartesianCoordDouble> || cuda::std::is_same_v<T, CartesianCoordBf16>;

template <typename T>
concept PolarCoord = cuda::std::is_same_v<T, PolarCoordFloat> || cuda::std::is_same_v<T, PolarCoordDouble> ||
                     cuda::std::is_same_v<T, PolarCoordBf16>;

template <typename T>
concept Coord = CartesianCoord<T> || PolarCoord<T>;

enum class KernelType {
  NAIVE,
  OPTIMIZED,
};

int parse_args(int argc, char* argv[], Args& args) {
  argparse::ArgumentParser program("coordinates");

  program.add_argument("--N")
      .help("The number of points")
      .scan<'i', int>()
      .default_value(1 << 24)  // ~16M points
      .store_into(args.N);
  program.add_argument("--block-size")
      .help("The block size")
      .scan<'i', int>()
      .default_value(64)
      .store_into(args.block_size);
  program.add_argument("--precision")
      .help("The precision: single, double, bf16")
      .choices("single", "double", "bf16")
      .default_value("single")
      .store_into(args.precision);
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
  printf("  Precision: %s\n", args.precision.c_str());
  return 0;
}


template <typename CartesianCoord>
std::vector<CartesianCoord> generate_cartesian_coordinates(int N) {
  std::vector<CartesianCoord> cartesian_coordinates(N, CartesianCoord(0.0f, 0.0f));
  std::default_random_engine generator(786);
  std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
  for (int i = 0; i < N; i++) {
    float x = distribution(generator);
    float y = distribution(generator);
    cartesian_coordinates[i] = CartesianCoord(x, y);
  }
  return cartesian_coordinates;
}


template <typename CartesianCoord, typename PolarCoord>
std::vector<PolarCoord> cartesian_to_polar_cpu(const std::vector<CartesianCoord>& cartesian_coordinates) {
  Timer timer("cartesian_to_polar_cpu");
  std::vector<PolarCoord> polar_coordinates(cartesian_coordinates.size(), PolarCoord(0.0f, 0.0f));
  std::transform(cartesian_coordinates.begin(), cartesian_coordinates.end(), polar_coordinates.begin(),
                 [](const CartesianCoord& cartesian_coordinate) -> PolarCoord {
                   float x = static_cast<float>(cartesian_coordinate.x);
                   float y = static_cast<float>(cartesian_coordinate.y);
                   float r = sqrt(x * x + y * y);
                   float theta = atan2(y, x);
                   return PolarCoord(r, theta);
                 });
  return polar_coordinates;
}


template <typename PolarCoord>
bool verify_results(const std::vector<PolarCoord>& polar_coordinates,
                    const std::vector<PolarCoord>& polar_coordinates_ref, double tolerance = 1e-4) {
  auto ret = std::ranges::equal(
      polar_coordinates, polar_coordinates_ref, [tolerance](const PolarCoord& a, const PolarCoord& b) -> bool {
        float r = a.x;
        float theta = a.y;
        float r_ref = b.x;
        float theta_ref = b.y;
        auto ret = std::fabs(r - r_ref) < tolerance && std::fabs(theta - theta_ref) < tolerance;
        if (!ret) {
          printf("Divergence! r: %5.4f != %5.4f (Diff %5.4f), theta: %5.4f != %5.4f (Diff %5.4f)", r_ref, r,
                 std::fabs(r - r_ref), theta_ref, theta, std::fabs(theta - theta_ref));
        }
        return ret;
      });
  printf("Results verified: %s\n", ret ? "true" : "false");
  return ret;
}

template <typename CartesianCoord, typename PolarCoord>
__global__ void cartesian_to_polar_kernel_0(CartesianCoord* _cartesian_coordinates, PolarCoord* d_polar_coordinates,
                                            int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoord coord = _cartesian_coordinates[idx];
    float x = static_cast<float>(coord.x);
    float y = static_cast<float>(coord.y);
    float r = sqrt(x * x + y * y);
    float theta = atan2(y, x);
    PolarCoord polar_coord = PolarCoord(r, theta);
    d_polar_coordinates[idx] = polar_coord;
  }
}


template <typename CartesianCoord, typename PolarCoord>
__global__ void cartesian_to_polar_kernel_1(CartesianCoord* __restrict__ _cartesian_coordinates,
                                            PolarCoord* __restrict__ _polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    CartesianCoord coord = _cartesian_coordinates[idx];
    float r = hypotf(coord.x, coord.y);
    float theta = atan2f(coord.y, coord.x);
    PolarCoord polar_coord = PolarCoord(r, theta);
    _polar_coordinates[idx] = polar_coord;
  }
}


template <typename CartesianCoord, typename PolarCoord>
void launch_kernel(int kernel_num, CartesianCoord* d_cartesian_coordinates, PolarCoord* d_polar_coordinates, int N,
                   int block_size, CudaStream& streamWrapper) {
  Timer timer(std::format("launch kernel no. {}", kernel_num));
  auto recorder = streamWrapper.record(std::format("kernel no. {}", kernel_num));
  dim3 dimBlock(block_size);
  dim3 dimGrid((N + block_size - 1) / block_size);

  switch (kernel_num) {
    case 0: {
      cartesian_to_polar_kernel_0<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(d_cartesian_coordinates,
                                                                                  d_polar_coordinates, N);
      break;
    }
    case 1: {
      cartesian_to_polar_kernel_1<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(d_cartesian_coordinates,
                                                                                  d_polar_coordinates, N);
      break;
    }
    default: {
      throw std::runtime_error(std::format("Invalid kernel number: {}", kernel_num));
    }
  }
  cudaCheck(cudaGetLastError());
  recorder.close();
}


template <typename CartesianCoord, typename PolarCoord>
int run_test(const Args& args, double tolerance) {
  int ret = 0;
  auto cartesian_coordinates = generate_cartesian_coordinates<CartesianCoord>(args.N);
  auto polar_coordinates_ref = cartesian_to_polar_cpu<CartesianCoord, PolarCoord>(cartesian_coordinates);
  auto polar_coordinates = std::vector<PolarCoord>(args.N, PolarCoord(0.0f, 0.0f));

  auto streamWrapper = CudaStream();
  cudaStream_t stream = streamWrapper.stream;
  auto d_cartesian_coordinates = make_cuda_unique<CartesianCoord>(args.N);
  cudaCheck(cudaMemcpyAsync(d_cartesian_coordinates.get(), cartesian_coordinates.data(),
                            args.N * sizeof(CartesianCoord), cudaMemcpyHostToDevice, streamWrapper.stream));
  auto d_polar_coordinates = make_cuda_unique<PolarCoord>(args.N);

  for (int kernel_num = 0; kernel_num < 2; kernel_num++) {
    launch_kernel(kernel_num, d_cartesian_coordinates.get(), d_polar_coordinates.get(), args.N, args.block_size,
                  streamWrapper);
    cudaCheck(cudaMemcpyAsync(polar_coordinates.data(), d_polar_coordinates.get(), args.N * sizeof(PolarCoord),
                              cudaMemcpyDeviceToHost, streamWrapper.stream));
    cudaCheck(cudaStreamSynchronize(streamWrapper.stream));
    ret |= verify_results<PolarCoord>(polar_coordinates, polar_coordinates_ref, tolerance);
  }
  return ret;
}


int main(int argc, char* argv[]) {
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  Args args;
  if (parse_args(argc, argv, args) != 0) {
    return 1;
  }

  if (args.precision == "single") {
    return run_test<CartesianCoordFloat, PolarCoordFloat>(args, 1e-4);
  } else if (args.precision == "double") {
    return run_test<CartesianCoordDouble, PolarCoordDouble>(args, 1e-6);
  } else if (args.precision == "bf16") {
    return run_test<CartesianCoordBf16, PolarCoordBf16>(args, 1e-2);
  } else {
    printf("Invalid precision: %s\n", args.precision.c_str());
    return 1;
  }
}