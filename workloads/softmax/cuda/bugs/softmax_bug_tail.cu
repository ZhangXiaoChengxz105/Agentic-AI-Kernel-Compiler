// BUG (tiling/boundary): the reduction loops only visit full BLOCK-sized chunks, so the last
// N % BLOCK elements are left out of max and sum. Invisible when N is a multiple of BLOCK.
// Category: tiling. Expected: VOLTA NOT-EQUIV only if checked with N % BLOCK != 0.
// Injected from softmax_opt.cu.
//
// Layout : x, y are row-major [rows, N] float32.
// Launch : grid = rows, block = BLOCK (=128, must be a power of two)
// Works for any N >= 1 (strided loops handle N not divisible by BLOCK).
#include "common.cuh"  // compile with -I <workspace>/kernels

#ifndef BLOCK
#define BLOCK 128
#endif

extern "C" __global__ void softmax_bug_tail(const float *__restrict__ x, float *__restrict__ y, int N) {
  __shared__ float red[BLOCK];
  const int tid = threadIdx.x;
  const float *xr = x + (size_t)blockIdx.x * N;
  float *yr = y + (size_t)blockIdx.x * N;

  // 1) row max: per-thread partial max, then tree reduction
  float m = -INFINITY;
  for (int i = tid; i < (N / BLOCK) * BLOCK; i += BLOCK) m = fmaxf(m, xr[i]);  // BUG: tail dropped
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
  for (int i = tid; i < (N / BLOCK) * BLOCK; i += BLOCK) sum += EXP(xr[i] - m);  // BUG: tail dropped
  red[tid] = sum;
  __syncthreads();
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / red[0];

  // 3) normalize (multiply by reciprocal instead of dividing)
  for (int i = tid; i < N; i += BLOCK) yr[i] = EXP(xr[i] - m) * inv;
}
