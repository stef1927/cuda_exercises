#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

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

struct PolarCoord {
  float r;      // radius
  float theta;  // angle in radians
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


std::vector<float2> generate_cartesian_coordinates(int N) {
  Timer timer("generate_cartesian_coordinates");
  std::vector<float2> cartesian_coordinates;
  cartesian_coordinates.reserve(N);
  std::default_random_engine generator(786);
  std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
  for (int i = 0; i < N; i++) {
    cartesian_coordinates.push_back(make_float2(distribution(generator), distribution(generator)));
  }
  return cartesian_coordinates;
}

std::vector<PolarCoord> cartesian_to_polar_cpu(const std::vector<float2>& cartesian_coordinates) {
  Timer timer("cartesian_to_polar_cpu");
  std::vector<PolarCoord> polar_coordinates(cartesian_coordinates.size(), PolarCoord{0.0f, 0.0f});
  std::transform(cartesian_coordinates.begin(), cartesian_coordinates.end(), polar_coordinates.begin(),
                 [](const float2& cartesian_coordinate) -> PolarCoord {
                   float r = sqrt(cartesian_coordinate.x * cartesian_coordinate.x +
                                  cartesian_coordinate.y * cartesian_coordinate.y);
                   float theta = atan2(cartesian_coordinate.y, cartesian_coordinate.x);
                   return PolarCoord{r, theta};
                 });
  return polar_coordinates;
}


bool verify_results(const std::vector<PolarCoord>& polar_coordinates,
                    const std::vector<PolarCoord>& polar_coordinates_ref, double tolerance = 1e-4) {
  auto ret = std::ranges::equal(
      polar_coordinates, polar_coordinates_ref, [tolerance](const PolarCoord& a, const PolarCoord& b) -> bool {
        auto ret = std::fabs(a.r - b.r) < tolerance && std::fabs(a.theta - b.theta) < tolerance;
        if (!ret) {
          printf("Divergence! r: %5.2f != %5.2f (Diff %5.2f), theta: %5.2f != %5.2f (Diff %5.2f)", b.r, a.r,
                 std::fabs(a.r - b.r), b.theta, a.theta, std::fabs(a.theta - b.theta));
        }
        return ret;
      });
  printf("Results verified: %s\n", ret ? "true" : "false");
  return ret;
}


__global__ void cartesian_to_polar_kernel_naive(float2* d_cartesian_coordinates, PolarCoord* d_polar_coordinates,
                                                int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_polar_coordinates[idx].r = sqrt(d_cartesian_coordinates[idx].x * d_cartesian_coordinates[idx].x +
                                      d_cartesian_coordinates[idx].y * d_cartesian_coordinates[idx].y);
    d_polar_coordinates[idx].theta = atan2(d_cartesian_coordinates[idx].y, d_cartesian_coordinates[idx].x);
  }
}

__global__ void cartesian_to_polar_kernel_optimized(float2* __restrict__ d_cartesian_coordinates,
                                                    PolarCoord* __restrict__ d_polar_coordinates, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float2 coord = d_cartesian_coordinates[idx];
    PolarCoord result;
    result.r = hypotf(coord.x, coord.y);
    result.theta = atan2f(coord.y, coord.x);
    d_polar_coordinates[idx] = result;
  }
}

void launch_kernel(float2* d_cartesian_coordinates, PolarCoord* d_polar_coordinates, int N, CudaStream& streamWrapper,
                   int block_size, KernelType kernelType) {
  Timer timer("cartesian_to_polar_kernel");
  CudaEventRecorder recorder = streamWrapper.record("cartesian_to_polar_kernel");
  dim3 dimBlock(block_size);
  dim3 dimGrid((N + block_size - 1) / block_size);

  switch (kernelType) {
    case KernelType::NAIVE: {
      cartesian_to_polar_kernel_naive<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(d_cartesian_coordinates,
                                                                                      d_polar_coordinates, N);
      break;
    }
    case KernelType::OPTIMIZED: {
      cartesian_to_polar_kernel_optimized<<<dimGrid, dimBlock, 0, streamWrapper.stream>>>(d_cartesian_coordinates,
                                                                                          d_polar_coordinates, N);
      break;
    }
    default: {
      throw std::runtime_error("Invalid kernel type");
    }
  }
  cudaCheck(cudaGetLastError());
}

int main(int argc, char* argv[]) {
  cudaDeviceProp deviceProp = getDeviceProperties(0, true);
  Args args;
  if (parse_args(argc, argv, args) != 0) {
    return 1;
  }

  auto cartesian_coordinates = generate_cartesian_coordinates(args.N);
  auto polar_coordinates = std::vector<PolarCoord>(args.N, PolarCoord{0.0f, 0.0f});
  auto polar_coordinates_ref = cartesian_to_polar_cpu(cartesian_coordinates);

  auto streamWrapper = CudaStream();
  cudaStream_t stream = streamWrapper.stream;
  auto d_cartesian_coordinates = make_cuda_unique<float2>(args.N);
  cudaCheck(cudaMemcpyAsync(d_cartesian_coordinates.get(), cartesian_coordinates.data(), args.N * sizeof(float2),
                            cudaMemcpyHostToDevice, streamWrapper.stream));
  auto d_polar_coordinates = make_cuda_unique<PolarCoord>(args.N);

  launch_kernel(d_cartesian_coordinates.get(), d_polar_coordinates.get(), args.N, streamWrapper, args.block_size,
                KernelType::NAIVE);

  cudaCheck(cudaMemcpyAsync(polar_coordinates.data(), d_polar_coordinates.get(), args.N * sizeof(PolarCoord),
                            cudaMemcpyDeviceToHost, streamWrapper.stream));
  cudaCheck(cudaStreamSynchronize(streamWrapper.stream));

  int ret = verify_results(polar_coordinates, polar_coordinates_ref);

  launch_kernel(d_cartesian_coordinates.get(), d_polar_coordinates.get(), args.N, streamWrapper, args.block_size,
                KernelType::OPTIMIZED);

  cudaCheck(cudaMemcpyAsync(polar_coordinates.data(), d_polar_coordinates.get(), args.N * sizeof(PolarCoord),
                            cudaMemcpyDeviceToHost, streamWrapper.stream));
  cudaCheck(cudaStreamSynchronize(streamWrapper.stream));

  ret |= verify_results(polar_coordinates, polar_coordinates_ref);

  return ret;
}