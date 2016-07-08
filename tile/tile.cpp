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
  hc::array_view<Point,2> tileIDs(globalExtent);

  hc::tiled_extent<2> tileExtent = globalExtent.tile(TILE_Y, TILE_X);
  hc::parallel_for_each(tileExtent, [=](hc::tiled_index<2> tidx) [[hc]] {
    tileIDs[tidx.global].x = tidx.local[1];
    tileIDs[tidx.global].y = tidx.local[0];
  });

  for (int j = 0; j < N_Y; j++) {
    for (int i = 0; i < N_X; i++) {
      printf("(%2d,%2d) ", tileIDs(j,i).x, tileIDs(j,i).y);
    }
    printf("\n");
  }

  return 0;
}
