
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

// header file for the hc API
#include <hc.hpp>

using namespace hc;

#define N  1024

int main() {

  const float a = 100.0f;
  float x[N];
  float y[N];

  // initialize the input data
  std::default_random_engine random_gen;
  std::uniform_real_distribution<float> distribution(-N, N);
  std::generate_n(x, N, [&]() { return distribution(random_gen); });
  std::generate_n(y, N, [&]() { return distribution(random_gen); });

  // make a copy of for the GPU implementation 
  float y_gpu[N];
  std::copy_n(y, N, y_gpu);

  // CPU implementation of saxpy
  for (int i = 0; i < N; i++) {
    y[i] = a * x[i] + y[i];
  }

  // wrap the data buffer around with an array_view
  // to let the hcc runtime to manage the data transfer
  array_view<float, 1> av_x(N, x);
  array_view<float, 1> av_y(N, y_gpu);

  // launch a GPU kernel to compute the saxpy in parallel 
  completion_future future;
  future = parallel_for_each(extent<1>(N), [=](index<1> i) __attribute((hc)) {
    av_y[i] = a * av_x[i] + av_y[i];
  });

  // the GPU kernel is launched asynchronously, wait for it to finish
  future.wait();

  // verify the results
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (fabs(y[i] - av_y[i]) > fabs(y[i] * 0.0001f))
      errors++;
  }
  std::cout << errors << " errors" << std::endl;

  return errors;
}
