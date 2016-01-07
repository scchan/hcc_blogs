
#include <cstdio>

// header file for the hc API
#include <hc.hpp>

void print_global_id(const int extent) {

  printf("\n%s: \n", __FUNCTION__);

  // global extent of the compute grid
  hc::extent<1> global_extent(extent);

  hc::array_view<int, 1> global_id(global_extent);

  hc::parallel_for_each(global_extent
                      , [=](hc::index<1> idx) [[hc]] {
    // get the global workitem ID of dimension 0
    // and store it into the global_id buffer
    global_id[idx] = idx[0];
  });

  // print the ID of each of the workitems
  for (int i = 0; i < extent; i++) {
    printf("global_id[%d]: %4d\n", i , global_id[i]);
  }
}

void print_global_local_tile_id(const int extent, const int tile_extent) {

  printf("\n%s: \n", __FUNCTION__);

  // global extent of the compute grid
  hc::extent<1> global_extent(extent);
  // to specify a group size, create an tiled_extent through the global extent
  hc::tiled_extent<1> t_extent = global_extent.tile(tile_extent);

  hc::array_view<int, 1> global_id(global_extent);
  hc::array_view<int, 1> tiled_id(global_extent);
  hc::array_view<int, 1> group_id(global_extent);

  // note that 
  hc::parallel_for_each(t_extent
                      , [=](hc::tiled_index<1> idx) [[hc]] {

    // get the global workitem ID 
    global_id[idx.global] = idx.global[0];

    // get the local workitem ID
    tiled_id[idx.global]  = idx.local[0];

    // get the ID of the tile/workgroup that this workitem belongs to
    group_id[idx.global]  = idx.tile[0];
  });

  for (int i = 0; i < extent; i++) {
    printf("global_id[%d]: %4d\t", i , global_id[i]);
    printf("tiled_id[%d]: %4d\t", i , tiled_id[i]);
    printf("group_id[%d]: %4d\n", i, group_id[i]);
  }
}


void group_left_rotate(const int extent, const int left_rotate) {

  printf("\n%s: \n", __FUNCTION__);

  hc::extent<1> global_extent(extent);
#define TILE_SIZE 64
  hc::tiled_extent<1> t_extent = global_extent.tile(TILE_SIZE);

  hc::array_view<int, 1> global_id(global_extent);
  hc::array_view<int, 1> tiled_id(global_extent);
  hc::array_view<int, 1> group_id(global_extent);
  hc::array_view<int, 1> result(global_extent);

  hc::parallel_for_each(t_extent
                      , [=](hc::tiled_index<1> idx) [[hc]] {

    // declare an tile_static array that is shared among 
    // workitems within the same tile
    tile_static int shared[TILE_SIZE];

    global_id[idx.global] = idx.global[0];
    tiled_id[idx.global]  = idx.local[0];
    group_id[idx.global]  = idx.tile[0];

    int data = idx.tile[0] * 1000 + idx.local[0];
    int local = idx.local[0];

    // using the local ID as index, store the value
    // computed by the current thread into the tiled_static array
    shared[local] = data;

    // use a barrier is needed to synchronize the data 
    // tiled_static
    idx.barrier.wait();

    // to achieve a rotate, load a value computed
    // by another workitem within the group from the
    // tiled_static array
    data = shared[(local+left_rotate)%TILE_SIZE];
    result[idx.global] = data;
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
  group_left_rotate(256, 4);

  return 0;
}
