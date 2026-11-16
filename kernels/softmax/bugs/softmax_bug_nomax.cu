// BUG (numerical): the max is not subtracted before exp. Mathematically identical
// (softmax is shift-invariant), but exp(x) overflows to inf in float32 once x > ~88.7.
// Category: numerical. Expected: VOLTA cannot see it (real-number semantics); only GPU tests
// with large-magnitude inputs catch it.
// Injected from softmax_opt.cu.
//
// Layout : x, y are row-major [rows, N] float32.
// Launch : grid = rows, block = BLOCK (=128, must be a power of two)
// Works for any N >= 1 (strided loops handle N not divisible by BLOCK).
#include "common.cuh"  // compile with -I <workspace>/kernels

#ifndef BLOCK
#define BLOCK 128
#endif

extern "C" __global__ void softmax_bug_nomax(const float *__restrict__ x, float *__restrict__ y, int N) {
  __shared__ float red[BLOCK];
  const int tid = threadIdx.x;
  const float *xr = x + (size_t)blockIdx.x * N;
  float *yr = y + (size_t)blockIdx.x * N;

  // 1) row max: per-thread partial max, then tree reduction
  float m = -INFINITY;
  for (int i = tid; i < N; i += BLOCK) m = fmaxf(m, xr[i]);
  red[tid] = m;
  __syncthreads();
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();  // everyone has read red[0] before it is reused below

  // 2) row sum of exp(x - m)
  float sum = 0.0f;
  for (int i = tid; i < N; i += BLOCK) sum += EXP(xr[i]);  // BUG: no max subtraction
  red[tid] = sum;
  __syncthreads();
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / red[0];

  // 3) normalize (multiply by reciprocal instead of dividing)
  for (int i = tid; i < N; i += BLOCK) yr[i] = EXP(xr[i]) * inv;  // BUG: no max subtraction
}
