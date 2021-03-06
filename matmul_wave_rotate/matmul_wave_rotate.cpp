#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <hc.hpp>

//#define DEBUG 1

constexpr int RAND_N = 10;

void printMatrix(std::vector<int>& mat, const int M, const int N) {
  for (int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      printf("%d,", mat[j*N+i]);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {

  constexpr int M_A = 128;
  constexpr int N_A = 256;

  constexpr int M_B = N_A;
  constexpr int N_B = 512;

  constexpr int M_C = M_A;
  constexpr int N_C = N_B;

  constexpr int wave_size = 64;
  static_assert(M_C%wave_size == 0,
                "The number of columns in Matrix A needs to be a multiple of 64");

  std::vector<int> matA(M_A * N_A);
  std::vector<int> matB(M_B * N_B);
  std::vector<int> matC(M_C * N_C);
  std::vector<int> matC_gpu(M_C * N_C);

  // initialize the input data
  std::default_random_engine random_gen;
  std::uniform_int_distribution<int> distribution(0, RAND_N);
  auto gen = std::bind(distribution, random_gen);
  std::generate(matA.begin(), matA.end(), gen);
  std::generate(matB.begin(), matB.end(), gen);

  // compute the dot product on the host
  for (int j = 0; j < M_C; j++) {
    for (int i = 0; i < N_C; i++) {
      int p = 0;
      for (int n = 0; n < N_A; n++) {
        int vA = matA[j * N_A + n];
        int vB = matB[n * N_B + i];
        p += vA * vB;
      }
      matC[j * N_C + i] = p;
    }
  }

  // create 2D array_views to present MxN matrices
  hc::array_view<int, 2> av_mat_A(M_A, N_A, matA);
  hc::array_view<int, 2> av_mat_B(M_B, N_B, matB);
  hc::array_view<int, 2> av_mat_C(hc::extent<2>(M_C, N_C), matC_gpu);

  // Launch a MxN kernel with each work-item computing one element in the result matrix.
  // Use a workgroup size equal to one wavefront.  Each workgroup would compute 
  // 64 adjacent elements on the same row in the output matrix
  hc::tiled_extent<2> t_extent = av_mat_C.get_extent().tile(1,wave_size);
  
  hc::parallel_for_each(t_extent, [=](hc::tiled_index<2> tidx) [[hc]] {
    int p = 0;
    for (int i = 0; i < N_A; i+=wave_size) {

      // Each workitem within a group would load a different input element from the matrix A.
      // Then after each iteration of the loop, we pass the data from matrix A
      // to the adjacent workitem using a rotate to share that data within a group.
      // After repeating this 64 times (size of a wave), the workgroup will move on to 
      // the next block of 64 elements from matrix A. 
      int vA = av_mat_A(tidx.global[0], i + tidx.local[1]);
      for (int j = 0; j < wave_size; j++) {
        int vB = av_mat_B(i + ((tidx.local[1] + j)%wave_size), tidx.global[1]);
        p += vA * vB;
        vA = hc::__amdgcn_wave_rl1(vA);
      }
    }
    av_mat_C(tidx.global) = p;
  });

  // synchronizes the results, which copies the data on the GPU
  // back to vector on the host
  av_mat_C.synchronize();

#ifdef DEBUG
  printf("matrix A:\n");
  printMatrix(matA, M_A, N_A);

  printf("matrix B:\n");
  printMatrix(matB, M_B, N_B);

  printf("matrix C:\n");
  printMatrix(matC, M_C, N_C);

  printf("matrix C (GPU):\n");
  printMatrix(matC_gpu, M_C, N_C);
#endif

  bool verify = std::equal(matC.begin(), matC.end(), matC_gpu.begin());
  printf("%s!\n", verify?"passed":"failed");

  return 0;
}
