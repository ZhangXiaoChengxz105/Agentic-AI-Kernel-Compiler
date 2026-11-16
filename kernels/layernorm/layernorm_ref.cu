// LayerNorm over the last dim, REFERENCE version: one thread per row, two-pass mean/variance.
//
// Layout : x, y row-major [rows, N] float32; w (gamma), b (beta) are [N] float32.
// Launch : grid = rows, block = 1.
// Math   : mean = sum(x)/N, var = sum((x-mean)^2)/N (biased, like torch), y = (x-mean)*rsqrt(var+eps)*w + b
#include "common.cuh"  // compile with -I <workspace>/kernels

extern "C" __global__ void layernorm_ref(const float *__restrict__ x, const float *__restrict__ w,
                                         const float *__restrict__ b, float *__restrict__ y, int N,
                                         float eps) {
  const float *xr = x + (size_t)blockIdx.x * N;
  float *yr = y + (size_t)blockIdx.x * N;

  float s = 0.0f;
  for (int i = 0; i < N; ++i) s += xr[i];
  const float mean = s / N;

  float v = 0.0f;
  for (int i = 0; i < N; ++i) {
    const float d = xr[i] - mean;
    v += d * d;
  }
  const float rstd = rsqrtf(v / N + eps);

  for (int i = 0; i < N; ++i) yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}
