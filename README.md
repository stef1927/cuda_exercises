# Parallel programming practice

A collection of CUDA and OpenMP programming exercises. Based on the following sources:

* NVIDIA [CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/master)
* NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
* *Programming Massively Parallel Processors, 4th Edition*, by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj, Published by Morgan Kaufmann

## Requirements

* NVIDIA HPC SDK for compiling the cpu examples, CUDA runtime toolkit (included in SDK) for gpu examples.

## Credits

* argparse.hpp taken from https://github.com/p-ranav/argparse
* sources listed above

## Building the Examples

You may need to change the paths in .clangd for the linter. 
Ensure nvcc and nvc++ can be found or edit the Makefile. For profiling, nsys, ncu, and vtune are also referenced from the Makefile.

All examples can be built using the provided Makefile with the following command:

```bash
make <example_file_name>
```

For example, to build the matrix multiplication exercise:

```bash
make matrix_mul
```

All compiled binaries are output to the `build` directory.

## Running the Examples

Each binary can be launched without arguments to use default parameters, or with the `--help` flag to display all supported command-line arguments and options.

## Matrix Multiplication

This exercise demonstrates multiple implementations of matrix multiplication with varying levels of optimization:

* **CUDA Naive Implementation**: A straightforward approach that reads data directly from global memory, with each thread computing one output element of the result matrix. This serves as a baseline for performance comparison.
* **CUDA Tiled Implementation**: An optimized version that improves upon the naive approach by loading matrix tiles into shared memory, significantly reducing global memory accesses and improving memory bandwidth utilization.
* **OpenMP Naive Parallel**: A multi-threaded naive CPU implementation based on OpenMP.
* **OpenMP Tiled Parallel**: A multi-threaded tiled CPU implementation based on OpenMP.

Matrices can be access in row-major or coloumn-major format.

## Histogram

This exercise implements a 256-bin histogram computation using advanced GPU optimization techniques. The implementation features a private histogram per thread block with privatization through shared memory, atomic operations for concurrent updates, and thread coarsening to improve computational throughput. The GPU results are verified against a CPU serial implementation for correctness. The exercise also demonstrates the use of CUDA streams, events, and asynchronous memory operations for efficient host-device communication.

## Reduction

This exercise performs parallel reduction to calculate the sum of a large array of integers. The implementation demonstrates the use of CUDA Cooperative Groups, a flexible programming model for thread synchronization and coordination. Specifically, it showcases reduction operations over warp tile groups, leveraging hardware-level primitives for efficient intra-warp communication and computation.

## Scan

This exercise implements an inclusive prefix scan (cumulative sum) operation using two distinct approaches. The first approach utilizes the optimized implementation from the NVIDIA CUB library. The second approach demonstrates a custom kernel implementation using CTA (Cooperative Thread Array) collective functions to scan individual warps using hardware-accelerated instructions. The custom implementation employs a hierarchical strategy: first scanning within warps, then reducing across warps within a thread block to scan the entire block, and finally reducing block sums to enable scanning of arbitrarily long arrays.

## Stream Compaction

This exercise demonstrates parallel stream compaction, a fundamental operation that selects elements from an input array matching a predicate function and copies only those elements to a contiguous output array. The implementation compares three distinct approaches:

* **STL Serial**: A baseline single-threaded CPU implementation using standard library algorithms.
* **STL Parallel**: A multi-threaded CPU implementation leveraging C++ parallel execution policies (`std::execution::par`).
* **OpenMP Parallel**: A multi-threaded CPU implementation based on OpenMP.
* **GPU Three-Pass Kernel**: A GPU-accelerated implementation employing a three-stage pipeline: (1) predicate evaluation to generate indicator flags, (2) inclusive scan using the CUB library to compute output positions, and (3) parallel gather of selected elements to their final positions in the output array.
