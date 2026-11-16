// GEMM, REFERENCE version: C = A @ B, one thread per output element, no shared memory.
//
// Layout : A [M, K], B [K, N], C [M, N], row-major float32.
// Launch : block = (TILE, TILE) = (16, 16), grid = (ceil(N/16), ceil(M/16)).
//          CTA (bx, by) owns the output tile rows [16*by, 16*by+16) x cols [16*bx, 16*bx+16).
#include "common.cuh"  // compile with -I <workspace>/kernels

#ifndef TILE
#define TILE 16
#endif

extern "C" __global__ void gemm_ref(const float *__restrict__ A, const float *__restrict__ B,
                                    float *__restrict__ C, int M, int N, int K) {
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}
