
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

// header file for the hc API
#include <hc.hpp>
#include <hc_am.hpp>

#define N  (1024 * 500)

int main() {

  const float a = 100.0f;
  float host_x[N];
  float host_y[N];

  // initialize the input data
  std::default_random_engine random_gen;
  std::uniform_real_distribution<float> distribution(-N, N);
  std::generate_n(host_x, N, [&]() { return distribution(random_gen); });
  std::generate_n(host_y, N, [&]() { return distribution(random_gen); });

  // make a copy of for the GPU implementation 
  float host_result_y[N];
  std::copy_n(host_y, N, host_result_y);

  // CPU implementation of saxpy
  for (int i = 0; i < N; i++) {
    host_result_y[i] = a * host_x[i] + host_y[i];
  }

  // allocate GPU memory through am_alloc
  hc::accelerator acc;
  float* x = hc::am_alloc(N * sizeof(float), acc, 0);
  float* y = hc::am_alloc(N * sizeof(float), acc, 0);

  // copy the data from host to GPU
  hc::am_copy(x, host_x, N * sizeof(float));
  hc::am_copy(y, host_y, N * sizeof(float));

  // launch a GPU kernel to compute the saxpy in parallel
  hc::completion_future future_pfe;
  future_pfe = hc::parallel_for_each(hc::extent<1>(N)
                                  , [=](hc::index<1> ind) [[hc]] {
    int i = ind[0];
    y[i] = a * x[i] + y[i];
  });

  // wait for the kernel to complete
  future_pfe.wait();

  // copy the data from GPU to host
  hc::am_copy(host_y, y, N * sizeof(float));
   
  // verify the results
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (fabs(host_y[i] - host_result_y[i]) > fabs(host_result_y[i] * 0.0001f))
      errors++;
  }
  std::cout << errors << " errors" << std::endl;

  return errors;
}
