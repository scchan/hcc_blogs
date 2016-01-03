
#include <cstdio>

// header file for the hc API
#include <hc.hpp>

#define N  256

int main() {

  hc::array_view<int, 1> global_id(hc::extent<1>(N));
  hc::parallel_for_each(hc::extent<1>(N)
                      , [=](hc::index<1> i) [[hc]] {
    global_id[i[0]] = i[0];
  }).wait();
  for (int i = 0; i < N; i++) {
    printf("global_id[%d]: %d\n", i , global_id[i]);
  }


  hc::array_view<int, 1> tiled_id(hc::extent<1>(N));
  hc::array_view<int, 1> group_id(hc::extent<1>(N));

   

  return 0;
}
