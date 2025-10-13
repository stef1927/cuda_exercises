#ifndef CUDA_EXERCISES_CPP_UTILS_H
#define CUDA_EXERCISES_CPP_UTILS_H

#include <stdio.h>

#include <chrono>

class Timer {
 public:
  Timer(const std::string& name) : name(name) { start = std::chrono::high_resolution_clock::now(); }

  ~Timer() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Total time taken to perform %s: %f milliseconds\n", name.c_str(), duration.count());
  }

 private:
  std::string name;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::chrono::duration<double, std::milli> duration;
};

inline void print_vector(const char* name, const std::vector<int>& output_data) {
  printf("%s: [", name);
  for (int i = 0; i < output_data.size(); i++) {
    printf("%d ", output_data[i]);
  }
  printf("]\n");
}

#endif