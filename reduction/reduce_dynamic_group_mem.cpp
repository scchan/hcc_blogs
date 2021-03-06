#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <hc.hpp>

#define __GROUP__ __attribute__((address_space(3)))

void reduction(const int num, const int tileSize) {
  
  int num_threads = num / 2;

  std::vector<int> data(num);
 
  // initialize the input data with random values
  constexpr int RAND_N = 100;
  std::default_random_engine random_gen;
  std::uniform_int_distribution<int> distribution(-RAND_N, RAND_N);
  auto gen = std::bind(distribution, random_gen);
  std::generate(data.begin(), data.end(), gen);

  const hc::array_view<int,1> av_data(num, data);

  // initialize the reduction sum to zero
  hc::array_view<int,1> reduced(1);
  reduced[0] = 0;

  // define the grid size an the tile size
  hc::extent<1> globalExtent(num_threads);
  hc::tiled_extent<1> tiledExtent = globalExtent.tile(tileSize);
  
  tiledExtent.set_dynamic_group_segment_size(tileSize * sizeof(int));

  hc::parallel_for_each(tiledExtent, [=](hc::tiled_index<1> tidx) [[hc]] {

    // static allocation of group memory
    __GROUP__ int* partialSums = (__GROUP__ int*) hc::get_dynamic_group_segment_base_pointer();
    
    // load 2 values from global memory and calculate a partial sum
    int localSum = av_data(tidx.global) 
                   + av_data(tidx.global[0] + globalExtent[0]);

    // store the partial sum in local memory corresponding to the current thread
    int localID = tidx.local[0];
    partialSums[localID] = localSum;

    // synchronize the local memory between threads in the same tile
    tidx.barrier.wait_with_tile_static_memory_fence();

    // With each iteration, reduce the number of threads participating in parallel reduction by
    // half.  For the threads that are still active, load 2 values from the tile_static array
    // and store the partial sum back to the static_static array
    for (int w = tidx.tile_dim[0]/2; w > 1; w/=2) {

      // check whether the current thread still participates in the reduction
      if (localID < w) {
        localSum = partialSums[localID]
                   + partialSums[localID + w];

        partialSums[localID] = localSum;
      }

      // synchronize the local memory between threads in the same tile
      tidx.barrier.wait_with_tile_static_memory_fence();
    }

    // have thread #0 to combine the last 2 partial sums
    // and to add it to the global reduction sum using an atomic add
    if (localID == 0) {
      localSum = partialSums[0] + partialSums[1];
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
}


int main() {
  constexpr int NUM = 1024 * 1024;
  for (int i = 64; i <= 512; i*=2) {
    printf("tile size %d:",i);
    reduction(NUM, i);
  }
  return 0;
}
