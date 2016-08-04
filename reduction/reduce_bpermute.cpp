#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <hc.hpp>

int main() {

  constexpr int NUM = 1024 * 1024;
  
  constexpr int NUM_THREADS = NUM / 2;
  constexpr int TILE_SIZE = 64;

  std::vector<int> data(NUM);
 
  // initialize the input data with random values
  constexpr int RAND_N = 100;
  std::default_random_engine random_gen;
  std::uniform_int_distribution<int> distribution(-RAND_N, RAND_N);
  auto gen = std::bind(distribution, random_gen);
  std::generate(data.begin(), data.end(), gen);

  const hc::array_view<int,1> av_data(NUM, data);

  // initialize the reduction sum to zero
  hc::array_view<int,1> reduced(1);
  reduced[0] = 0;

  // define the grid size an the tile size
  hc::extent<1> globalExtent(NUM_THREADS);
  hc::tiled_extent<1> tiledExtent = globalExtent.tile(TILE_SIZE);

  hc::parallel_for_each(tiledExtent, [=](hc::tiled_index<1> tidx) [[hc]] {

    // load 2 values from global memory and calculate a partial sum
    int localSum = av_data(tidx.global) 
                   + av_data(tidx.global[0] + globalExtent[0]);

    // bpermute the partial sum from another
    // threads and add to the local partial sum.
    // With each iteration, reduce the thread distance by half until
    // it reaches zero
    for (int w = TILE_SIZE/2; w > 0; w/=2) {
      int neighbor = hc::__amdgcn_ds_bpermute((tidx.local[0] + w)<<2, localSum);
      localSum += neighbor;
    }

    // thread #0 holds the correct value
    int localID = tidx.local[0];
    if (localID == 0) {
      hc::atomic_fetch_add(&reduced[0], localSum);
    }
  });

  // calculate the reduction on the CPU and verify the result
  int hostReduced = std::accumulate(data.begin(), data.end(), 0);
  if (reduced[0] == hostReduced) {
    printf("passed\n");
  }
  else {
    printf("failed, expected=%d, actual=%d\n", hostReduced, reduced[0]);
  }

  return 0;
}
