#include <cstdio>
#include <vector>
#include <hc.hpp>

struct Point {
  int x;
  int y;
};


constexpr int N_X = 32;
constexpr int N_Y = 32;

constexpr int TILE_X = 16;
constexpr int TILE_Y = 4;

int main() {

  hc::extent<2> globalExtent(N_Y, N_X);

  // local thread IDs
  hc::array_view<Point,2> localIDs(globalExtent);

  // tile IDs
  hc::array_view<Point,2> tileIDs(globalExtent);


  hc::tiled_extent<2> tileExtent = globalExtent.tile(TILE_Y, TILE_X);
  hc::parallel_for_each(tileExtent, [=](hc::tiled_index<2> tidx) [[hc]] {

    // store the local thread ID within a tile
    localIDs[tidx.global].x = tidx.local[1];
    localIDs[tidx.global].y = tidx.local[0];

    // store the ID of this tile of threads
    tileIDs[tidx.global].x = tidx.tile[1];
    tileIDs[tidx.global].y = tidx.tile[0];
  });

  // print out the local IDs
  printf("Local IDs:\n");
  for (int j = 0; j < N_Y; j++) {
    for (int i = 0; i < N_X; i++) {
      printf("(%2d,%2d) ", localIDs(j,i).x, localIDs(j,i).y);
    }
    printf("\n");
  }

  // print out the tile IDs
  printf("\n\nTile IDs\n");
  for (int j = 0; j < N_Y; j++) {
    for (int i = 0; i < N_X; i++) {
      printf("(%2d,%2d) ", tileIDs(j,i).x, tileIDs(j,i).y);
    }
    printf("\n");
  }

  return 0;
}
