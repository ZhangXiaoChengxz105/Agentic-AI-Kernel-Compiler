// Row-wise softmax for sm_80 (A100).
// Launch: gridDim.x = rows, blockDim.x = 128 (one CTA per row).
//
// Verification fix: the previous version used the "online softmax" trick,
// rescaling the running sum with factors like EXP(m - m_new). That multiplies
// exponentials together, and products such as EXP(-200)*EXP(-200) blow up the
// exact-rational (i128) coefficients used by the formal verifier. This version
// performs three clean passes with NO products of exponentials:
//   1) reduce the row maximum M (fmaxf only, no EXP),
//   2) reduce L = sum_i EXP(x_i - M) (a plain sum of single exponentials),
//   3) write y_i = EXP(x_i - M) / L (direct division, matching the reference).
// All control flow depends only on threadIdx/blockIdx/blockDim/N.

#include "common.cuh"
#include <cfloat>

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid  = threadIdx.x;          // 0..127
    const int warp = tid >> 5;             // 0..3  (blockDim.x == 128 -> 4 warps)
    const int lane = tid & 31;

    const float* __restrict__ xr = x + (size_t)blockIdx.x * (size_t)N;
    float*       __restrict__ yr = y + (size_t)blockIdx.x * (size_t)N;

    __shared__ float smax[4];
    __shared__ float ssum[4];

    // ------------------------------------------------------------------
    // Pass 1: block-wide maximum M of the row (pure fmaxf, no exponentials).
    // 4 strided, fully coalesced loads per iteration.
    // ------------------------------------------------------------------
    float m = -FLT_MAX;
    int i = tid;
    while (N - i > 384) {
        const float v0 = xr[i];
        const float v1 = xr[i + 128];
        const float v2 = xr[i + 256];
        const float v3 = xr[i + 384];
        m = fmaxf(fmaxf(fmaxf(m, v0), v1), fmaxf(v2, v3));
        i += 512;
    }
    for (; i < N; i += 128)
        m = fmaxf(m, xr[i]);

    // warp-level max reduction (xor butterfly: every lane gets the result)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    if (lane == 0) smax[warp] = m;
    __syncthreads();

    // Global row max (every thread computes it; identical value for all).
    const float M = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(smax[2], smax[3]));

    // ------------------------------------------------------------------
    // Pass 2: block-wide sum L = sum_i EXP(x_i - M).
    // Only additions of single exponentials (arguments <= 0); no products.
    // ------------------------------------------------------------------
    float l = 0.0f;
    i = tid;
    while (N - i > 384) {
        const float e0 = EXP(xr[i]       - M);
        const float e1 = EXP(xr[i + 128] - M);
        const float e2 = EXP(xr[i + 256] - M);
        const float e3 = EXP(xr[i + 384] - M);
        l = (e0 + e1) + (e2 + l);
        i += 512;
    }
    for (; i < N; i += 128)
        l = l + EXP(xr[i] - M);

    // warp-level sum reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        l = l + __shfl_xor_sync(0xffffffffu, l, off);

    if (lane == 0) ssum[warp] = l;
    __syncthreads();

    // Total sum (L >= 1, since the max element contributes EXP(0) = 1).
    const float L = (ssum[0] + ssum[1]) + (ssum[2] + ssum[3]);

    // ------------------------------------------------------------------
    // Pass 3: y_i = EXP(x_i - M) / L  (second coalesced read of x).
    // Direct division matches the reference formula exactly; no reciprocal
    // is precomputed, so no extra rational operations are introduced.
    // ------------------------------------------------------------------
    i = tid;
    while (N - i > 384) {
        yr[i]       = EXP(xr[i]       - M) / L;
        yr[i + 128] = EXP(xr[i + 128] - M) / L;
        yr[i + 256] = EXP(xr[i + 256] - M) / L;
        yr[i + 384] = EXP(xr[i + 384] - M) / L;
        i += 512;
    }
    for (; i < N; i += 128)
        yr[i] = EXP(xr[i] - M) / L;
}
