#ifndef CUDA_EXERCISES_CPP_UTILS_H
#define CUDA_EXERCISES_CPP_UTILS_H

#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#include <chrono>
#include <string>
#include <vector>

class Timer {
 public:
  Timer(const char* name, int num_runs = 1, bool auto_start = true)
      : name(name), num_runs(num_runs), running(false), duration_millis(0.0f) {
    if (auto_start) {
      start();
    }
  }
  Timer(const std::string& name, int num_runs = 1, bool auto_start = true)
      : name(name), num_runs(num_runs), running(false), duration_millis(0.0f) {
    if (auto_start) {
      start();
    }
  }

  void start() {
    if (!running) {
      begin = std::chrono::steady_clock::now();
      running = true;
    }
  }

  void stop() {
    if (running) {
      end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> duration = end - begin;
      begin = end;
      duration_millis += duration.count();
      running = false;
    }
  }

  ~Timer() {
    stop();
    if (num_runs > 1) {
      printf("Total time taken to perform %d runs of %s: %f milliseconds\n", num_runs, name.c_str(), duration_millis);
      printf("Average time taken per run: %f milliseconds\n", duration_millis / num_runs);
    } else {
      printf("Time taken to perform %s: %f milliseconds\n", name.c_str(), duration_millis);
    }
  }

 private:
  std::string name;
  int num_runs;
  bool running;
  std::chrono::steady_clock::time_point begin;
  std::chrono::steady_clock::time_point end;
  double duration_millis;
};

template <typename T>
inline void print_vector(const char* name, const T* output_data, size_t size) {
  printf("%s: [", name);
  for (size_t i = 0; i < size; i++) {
    printf("%d ", output_data[i]);
  }
  printf("]\n");
}

template <typename T>
inline void print_vector(const char* name, const std::vector<T>& output_data) {
  return print_vector(name, output_data.data(), output_data.size());
}

#endif