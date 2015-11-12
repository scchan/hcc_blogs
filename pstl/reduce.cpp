#include <array_view>
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

// header file for the hc API
#include <hc.hpp>
#include "hc_am.hpp"

#define N  (1024 * 500)

int main() {

  float x[N];

  // initialize the input data
  std::default_random_engine random_gen;
  std::uniform_real_distribution<float> distribution(-N, N);
  std::generate_n(x, N, [&]() { return distribution(random_gen); });

  hc::accelerator_view acc_view = hc::accelerator().create_view();
  float* x_gpu = static_cast<float*>(am_alloc(N * sizeof(float), AM_EXPLICIT_SYNC, acc_view));
  am_copy(x_gpu, x, N * sizeof(float), acc_view);

  float r_gpu = std::experimental::parallel::reduce(std::experimental::parallel::par,
                                        x_gpu, x_gpu + N);
 
  float r_host = std::accumulate(x, x + N, 0.0f);

  if (fabs(r_host - r_gpu) > fabs(r_host * 0.0001f))
    std::cout << "Error: expected = " << r_host << " actual = " << r_gpu << std::endl;
  else
    std::cout << "Verified!" << std::endl;

  return 0;
}
