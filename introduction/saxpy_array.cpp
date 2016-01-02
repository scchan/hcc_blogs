
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <exception>

// header file for the hc API
#include <hc.hpp>

#define N  (1024 * 500)

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

  // allocate data buffers on the accelerator and copy the data over
  hc::array<float, 1> array_x(N);
  hc::completion_future future_x = hc::copy_async(x, x + N, array_x);

  hc::array<float, 1> array_y(N);
  hc::completion_future future_y = hc::copy_async(y_gpu, y_gpu + N, array_y);

  // wait for the copy operations to complete
  future_x.wait();
  future_y.wait();

  // launch a GPU kernel to compute the saxpy in parallel
  hc::completion_future future_pfe;
  future_pfe = hc::parallel_for_each(hc::extent<1>(N)
                      , [&](hc::index<1> i) [[hc]] {
    array_y[i] = a * array_x[i] + array_y[i];
  });

  // wait for the kernel to complete before copying results back
  // to the host
  future_pfe.wait();
  future_y = hc::copy_async(array_y, y_gpu);
  future_y.wait();

  // verify the results
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (fabs(y[i] - y_gpu[i]) > fabs(y[i] * 0.0001f))
      errors++;
  }
  std::cout << errors << " errors" << std::endl;

  return errors;
}
