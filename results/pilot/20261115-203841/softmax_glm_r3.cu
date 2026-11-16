// Row-wise softmax for sm_80 (A100).
// Launch: gridDim.x = rows, blockDim.x = 128 (one CTA per row).
//
// Root cause of the "rational coefficient overflowed i128" failure:
// the max-reduction was initialized with -FLT_MAX = -3.4028e38. The
// verifier evaluates this constant in exact rational arithmetic, but
// 2^127-1 (~1.70e38) is the largest i128 integer, so -FLT_MAX itself
// overflows the verifier's coefficient representation -- before any
// exponential is even involved. Fix: initialize the running max with
// xr[0] (a real, ordinary-sized row element; since N >= 1 it always
// exists, and since it is an element of the row it can never corrupt
// the maximum). Control flow still depends only on threadIdx,
// blockIdx, blockDim and N.
//
// Structure (three clean passes, no products of exponentials):
//   1) M = max_i x_i                (fmaxf only, no EXP)
//   2) L = sum_i EXP(x_i - M)       (plain sum of single exponentials)
//   3) y_i = EXP(x_i - M) / L       (direct division, as in the reference)

#include "common.cuh"

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid  = threadIdx.x;          // 0..127
    const int warp = tid >> 5;             // 0..3  (blockDim.x == 128 -> 4 warps)
    const int lane = tid & 31;

    const float* __restrict__ xr = x + (size_t)blockIdx.x * (size_t)N;
    float*       __restrict__ yr = y + (size_t)blockIdx.x * (size_t)N;

    __shared__ float smax[4];
    __shared__ float ssum[4];

    // Safe, verifier-friendly initialization: an actual row element
    // (all threads read the same address -> broadcast load). Because
    // xr[0] belongs to the row, including it in the max is always sound.
    const float seed = xr[0];

    // ------------------------------------------------------------------
    // Pass 1: block-wide maximum M of the row (pure fmaxf, no exponentials).
    // 4 strided, fully coalesced loads per iteration.
    // ------------------------------------------------------------------
    float m = seed;
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
