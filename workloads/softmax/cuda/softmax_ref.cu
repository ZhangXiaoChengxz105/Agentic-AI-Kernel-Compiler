// Row-wise softmax, REFERENCE version: obviously correct, not fast.
//
// Layout : x, y are row-major [rows, N] float32.
// Launch : grid = rows, block = 1   (one CTA per row, one thread does the whole row)
// Math   : y[r, i] = exp(x[r, i] - m) / sum_j exp(x[r, j] - m),  m = max_j x[r, j]
#include "common.cuh"  // compile with -I <workspace>/kernels

extern "C" __global__ void softmax_ref(const float *__restrict__ x, float *__restrict__ y, int N) {
  const float *xr = x + (size_t)blockIdx.x * N;
  float *yr = y + (size_t)blockIdx.x * N;

  float m = -INFINITY;
  for (int i = 0; i < N; ++i) m = fmaxf(m, xr[i]);

  float s = 0.0f;
  for (int i = 0; i < N; ++i) s += EXP(xr[i] - m);

  for (int i = 0; i < N; ++i) yr[i] = EXP(xr[i] - m) / s;
}
