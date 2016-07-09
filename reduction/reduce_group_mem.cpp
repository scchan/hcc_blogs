#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <hc.hpp>

int main() {

  constexpr int NUM = 1024 * 1024;
  //constexpr int NUM = 128;
  
  constexpr int NUM_THREADS = NUM / 2;
  constexpr int TILE_SIZE = 64;

  std::vector<int> data(NUM);
 
  // initialize the input data
  constexpr int RAND_N = 100;
  std::default_random_engine random_gen;
  std::uniform_int_distribution<int> distribution(-RAND_N, RAND_N);
  auto gen = std::bind(distribution, random_gen);
  std::generate(data.begin(), data.end(), gen);

  const hc::array_view<int,1> av_data(NUM, data);
  hc::array_view<int,1> reduced(1);
  reduced[0] = 0;

  hc::extent<1> globalExtent(NUM_THREADS);
  hc::tiled_extent<1> tiledExtent = globalExtent.tile(TILE_SIZE);

  hc::parallel_for_each(tiledExtent, [=](hc::tiled_index<1> tidx) [[hc]] {

    // static allocation of group memory
    tile_static int partialSums[TILE_SIZE];
    
    // load 2 values from global memory and calculate a partial sum
    int localSum = av_data(tidx.global) 
                   + av_data(tidx.global[0] + globalExtent[0]);

    // store the partial sum in local memory corresponding to the current thread
    int localID = tidx.local[0];
    partialSums[localID] = localSum;

    for (int w = tidx.tile_dim[0]/2; w > 1; w/=2) {

      // synchronize the local memory between threads in the same tile
      tidx.barrier.wait_with_tile_static_memory_fence();

      if (localID < w) {
        localSum = partialSums[localID]
                   + partialSums[localID + w];

        partialSums[localID] = localSum;
      }
    }

    // synchronize the local memory between threads in the same tile
    tidx.barrier.wait_with_tile_static_memory_fence();
    if (localID == 0) {
      localSum = partialSums[0] + partialSums[1];
      hc::atomic_fetch_add(&reduced[0], localSum);
    }
  });

  int hostReduced = std::accumulate(data.begin(), data.end(), 0);
  if (reduced[0] == hostReduced) {
    printf("passed\n");
  }
  else {
    printf("failed, expected=%d, actual=%d\n", hostReduced, reduced[0]);
  }

  return 0;
}
