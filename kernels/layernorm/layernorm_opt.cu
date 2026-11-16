// LayerNorm over the last dim, OPTIMIZED version: block-parallel two-pass with tree reductions.
//
// Layout : x, y row-major [rows, N] float32; w (gamma), b (beta) are [N] float32.
// Launch : grid = rows, block = BLOCK (=128, power of two). Works for any N >= 1.
#include "common.cuh"  // compile with -I <workspace>/kernels

#ifndef BLOCK
#define BLOCK 128
#endif

extern "C" __global__ void layernorm_opt(const float *__restrict__ x, const float *__restrict__ w,
                                         const float *__restrict__ b, float *__restrict__ y, int N,
                                         float eps) {
  __shared__ float red[BLOCK];
  const int tid = threadIdx.x;
  const float *xr = x + (size_t)blockIdx.x * N;
  float *yr = y + (size_t)blockIdx.x * N;

  // 1) mean
  float s = 0.0f;
  for (int i = tid; i < N; i += BLOCK) s += xr[i];
  red[tid] = s;
  __syncthreads();
  for (int k = BLOCK / 2; k > 0; k >>= 1) {
    if (tid < k) red[tid] += red[tid + k];
    __syncthreads();
  }
  const float mean = red[0] / N;
  __syncthreads();  // everyone has read red[0] before it is reused below

  // 2) variance (second pass over centered values)
  float v = 0.0f;
  for (int i = tid; i < N; i += BLOCK) {
    const float d = xr[i] - mean;
    v += d * d;
  }
  red[tid] = v;
  __syncthreads();
  for (int k = BLOCK / 2; k > 0; k >>= 1) {
    if (tid < k) red[tid] += red[tid + k];
    __syncthreads();
  }
  const float rstd = rsqrtf(red[0] / N + eps);

  // 3) normalize + affine
  for (int i = tid; i < N; i += BLOCK) yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}
