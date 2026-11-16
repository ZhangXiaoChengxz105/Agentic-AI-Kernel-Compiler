// BUG (indexing, off-by-one): the output guard uses <= instead of <, so threads with row == M or
// col == N write one row/column past the end of C. Only CTAs that touch the matrix edge are affected.
// Category: indexing. Expected: VOLTA catches it only if CTA 0 touches the edge (M or N < 16);
// otherwise EQUIV (bug hidden). Injected from gemm_opt.cu.
//
// Layout : A [M, K], B [K, N], C [M, N], row-major float32.
// Launch : block = (TILE, TILE) = (16, 16), grid = (ceil(N/16), ceil(M/16)).
// Each CTA walks K in TILE-wide steps: load a TILExTILE tile of A and of B into shared memory
// (zero-padding out-of-range elements), barrier, accumulate, barrier, next tile.
// Works for any M, N, K >= 1.
#include "common.cuh"  // compile with -I <workspace>/kernels

#ifndef TILE
#define TILE 16
#endif

extern "C" __global__ void gemm_bug_guard(const float *__restrict__ A, const float *__restrict__ B,
                                    float *__restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  const int ntiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < ntiles; ++t) {
    const int ka = t * TILE + tx;  // column of A loaded by this thread
    const int kb = t * TILE + ty;  // row of B loaded by this thread
    As[ty][tx] = (row < M && ka < K) ? A[row * K + ka] : 0.0f;
    Bs[ty][tx] = (kb < K && col < N) ? B[kb * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row <= M && col <= N) C[row * N + col] = acc;  // BUG: <= instead of <
}
