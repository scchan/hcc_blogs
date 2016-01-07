
#include <cstdio>

// header file for the hc API
#include <hc.hpp>

void print_global_id(const int extent) {

  hc::extent<1> global_extent(extent);
  hc::array_view<int, 1> global_id(global_extent);
  hc::parallel_for_each(global_extent
                      , [=](hc::index<1> i) [[hc]] {
    global_id[i] = i[0];
  });

  for (int i = 0; i < extent; i++) {
    printf("global_id[%d]: %4d\n", i , global_id[i]);
  }
}

void print_global_local_tile_id(const int extent, const int tile_extent) {

  hc::extent<1> global_extent(extent);

  hc::array_view<int, 1> global_id(global_extent);
  hc::array_view<int, 1> tiled_id(global_extent);
  hc::array_view<int, 1> group_id(global_extent);

  hc::tiled_extent<1> t_extent = global_extent.tile(tile_extent);

  hc::parallel_for_each(t_extent
                      , [=](hc::tiled_index<1> i) [[hc]] {
    int global = i.global[0];
    global_id[i] = i.global[0];
    tiled_id[i] = i.local[0];
    group_id[i] = i.tile[0];
  });

  for (int i = 0; i < extent; i++) {
    printf("global_id[%d]: %4d\t", i , global_id[i]);
    printf("tiled_id[%d]: %4d\t", i , tiled_id[i]);
    printf("group_id[%d]: %4d\n", i, group_id[i]);
  }
}


void group_rotate(const int extent) {

  hc::extent<1> global_extent(extent);

  hc::array_view<int, 1> global_id(global_extent);
  hc::array_view<int, 1> tiled_id(global_extent);
  hc::array_view<int, 1> group_id(global_extent);
  hc::array_view<int, 1> result(global_extent);

#define TILE_SIZE 64
  hc::tiled_extent<1> t_extent = global_extent.tile(TILE_SIZE);

  hc::parallel_for_each(t_extent
                      , [=](hc::tiled_index<1> i) [[hc]] {

    // declare a tile_static array that is shared among 
    // workitems within the same tile
    tile_static int shared[TILE_SIZE];

    int global = i.global[0];
    global_id[global] = i.global[0];
    tiled_id[global] = i.local[0];
    group_id[global] = i.tile[0];

    int data = i.tile[0] * 1000 + i.local[0];
    int local = i.local[0];

    shared[local] = data;
    i.barrier.wait();

    int rotate = 4;
    data = shared[(local+rotate)%TILE_SIZE];
    result[global] = data;

  });

  for (int i = 0; i < extent; i++) {
    printf("global_id[%d]: %4d\t", i , global_id[i]);
    printf("tiled_id[%d]: %4d\t", i , tiled_id[i]);
    printf("group_id[%d]: %4d\t", i, group_id[i]);
    printf("result[%d]: %4d\n", i, result[i]);
  }

}

int main() {
  
  // print the global IDs of all 256 workitems
  print_global_id(256);

  // print the global, local and tile IDs of all 256 workitems
  // with a tile size of 64 workitems
  print_global_local_tile_id(256, 64);

  // print out the IDs as above but with a tile size of 128 workitems
  print_global_local_tile_id(256, 128);

  // use tile static memory to perform a group rotate
  group_rotate(256);

  return 0;
}
