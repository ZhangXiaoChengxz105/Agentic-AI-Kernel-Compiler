#include "common.cuh"

// Row-wise softmax: one CTA (128 threads) per row.
// Numerically stable, works for any N >= 1, data-independent control flow.
extern "C" __global__ void softmax_glm(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int N)
{
    const int tid  = threadIdx.x;          // 0..127
    const int lane = tid & 31;
    const int warp = tid >> 5;             // 0..3 (blockDim.x == 128)
    const int nthreads = blockDim.x;       // 128

    const size_t base = (size_t)blockIdx.x * (size_t)N;
    const float* __restrict__ xr = x + base;
    float*       __restrict__ yr = y + base;

    __shared__ float smax[4];
    __shared__ float ssum[4];

    // ------------------------------------------------------------------
    // Pass 1: row maximum (tree reduction with warp shuffles)
    // ------------------------------------------------------------------
    float m = -INFINITY;
    for (int i = tid; i < N; i += nthreads)
        m = fmaxf(m, xr[i]);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    if (lane == 0) smax[warp] = m;
    __syncthreads();

    m = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(smax[2], smax[3]));

    // ------------------------------------------------------------------
    // Pass 2: e_i = EXP(x_i - max), stash in y, accumulate row sum
    // ------------------------------------------------------------------
    float s = 0.0f;
    for (int i = tid; i < N; i += nthreads) {
        float e = EXP(xr[i] - m);
        yr[i] = e;
        s += e;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);

    if (lane == 0) ssum[warp] = s;
    __syncthreads();

    s = ssum[0] + ssum[1] + ssum[2] + ssum[3];
    const float inv = 1.0f / s;   // s >= 1 (the max element contributes 1)

    // ------------------------------------------------------------------
    // Pass 3: normalize in place (y already holds the exponentials)
    // ------------------------------------------------------------------
    for (int i = tid; i < N; i += nthreads)
        yr[i] *= inv;
}
