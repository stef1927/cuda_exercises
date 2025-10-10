#ifndef CUDA_EXERCISES_CPP_UTILS_H
#define CUDA_EXERCISES_CPP_UTILS_H

#include <stdio.h>

#include <chrono>

class Timer {
 public:
  Timer(const char* name) : name(name) { start = std::chrono::high_resolution_clock::now(); }

  ~Timer() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Total time taken to perform %s: %f milliseconds\n", name, duration.count());
  }

 private:
  const char* name;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::chrono::duration<double, std::milli> duration;
};

#endif