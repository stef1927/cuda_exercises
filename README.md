# CUDA exercises

A collection of CUDA exercises developed by starting with the following sources:

* NVIDIA [CUDA samples](https://github.com/NVIDIA/cuda-samples/tree/master)
* NVIDIA [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
* Programming Massively Parallel Processors, 4th Edition, Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, Published by Morgan Kaufmann
* Command line parsing is taken from [ArgParse](https://github.com/p-ranav/argparse) directly, without modifications. This was done to avoid dependencies.

All examples can be built with: 

```
make <example_file_name>
```

For example:

```
make matrix_mul
```

All binaries are built into the `build` directory.

Launch the binary without arguments, or with `--help` for finding out the arguments supported.

Refer to the `Makefile` for the NCU profiling flags, or launch
profiling through the following make command:

```
make <example_file_name>_profile
```

for example:

```
make matrix_mul_profile
```

## Matrix multiplication

Different implementations of matrix multiplication:

* Naive, the naive implementation reading data from global memory and mapping one thread to every output element in the matrix.
* Tiled, improves the naive implementation by loading matrix tiles into shared memory.

## Histogram

Implements a 256-bin histogram using a GPU kernel with one private histogram per thread block, privatization via shared memory, atomic adds and thread coarsening. Verifies the histogram against a CPU serial implementation. Uses streams, CUDA events and async memory operations.

## Reduction

Calculates the sum of a large array of integers. Demonstrates using cooperative groups, including
reduction over a warp tile group.

## Scan

Performs an inclusive prefix scan (sum) using the implementation from the CUB library, and a kernel using the CTA collective functions
to scan individual warps with the hardware accelerated function, then reduce warps in a block to scan the entire block, and reduce block
sums to scan an arbitrarily long array.