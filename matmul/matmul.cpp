
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <hc.hpp>

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

//#define DEBUG 1

#ifndef DEBUG
  constexpr int M_A = 128;
  constexpr int N_A = 256;

  constexpr int M_B = N_A;
  constexpr int N_B = 512;

  constexpr int M_C = M_A;
  constexpr int N_C = N_B;
#else
  constexpr int M_A = 4;
  constexpr int N_A = 3;

  constexpr int M_B = N_A;
  constexpr int N_B = 2;

  constexpr int M_C = M_A;
  constexpr int N_C = N_B;
#endif

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

  hc::array_view<int, 2> av_mat_A(M_A, N_A, matA);
  hc::array_view<int, 2> av_mat_B(M_B, N_B, matB);
  hc::array_view<int, 2> av_mat_C(hc::extent<2>(M_C, N_C), matC_gpu);

  hc::parallel_for_each(av_mat_C.get_extent(), [=](hc::index<2> idx) [[hc]] {
    int p = 0;
    for (int n = 0; n < N_A; n++) {
      int vA = av_mat_A(idx[0], n);
      int vB = av_mat_B(n, idx[1]);
      p += vA * vB;
    }
    av_mat_C(idx) = p;
  });


  printf("matrix A:\n");
  printMatrix(matA, M_A, N_A);

  printf("matrix B:\n");
  printMatrix(matB, M_B, N_B);

  printf("matrix C:\n");
  printMatrix(matC, M_C, N_C);

  av_mat_C.synchronize();
  printf("matrix C (GPU):\n");
  printMatrix(matC_gpu, M_C, N_C);


  bool verify = std::equal(matC.begin(), matC.end(), matC_gpu.begin());
  printf("GPU results match with host results: %s\n", verify?"true":"false");

  return 0;
}
