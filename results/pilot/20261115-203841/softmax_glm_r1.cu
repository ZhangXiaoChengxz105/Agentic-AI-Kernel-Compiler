// Row-wise softmax for sm_80 (A100).
// Launch: gridDim.x = rows, blockDim.x = 128 (one CTA per row).
// - Numerically stable online (single-read) max/sum-of-exp pass, then a
//   normalize/store pass (x is read twice total, fully coalesced).
// - All control flow depends only on threadIdx/blockIdx/blockDim/N.
// - All exponentials go through the EXP() macro from common.cuh.

#include "common.cuh"
#include <cfloat>

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid  = threadIdx.x;          // 0..127
    const int warp = tid >> 5;             // 0..3  (blockDim.x == 128 -> 4 warps)
    const int lane = tid & 31;

    const float* __restrict__ xr = x + (size_t)blockIdx.x * (size_t)N;
    float*       __restrict__ yr = y + (size_t)blockIdx.x * (size_t)N;

    // ------------------------------------------------------------------
    // Pass 1: streaming online softmax over this thread's strided elements
    //         m : running max of the elements seen so far
    //         l : running sum of exp(v - m)
    // Note: m - m_new <= 0 and v - m_new <= 0 always, so every EXP() arg
    //       is <= 0: no overflow, and exp underflow is harmless.
    // ------------------------------------------------------------------
    float m = -FLT_MAX;
    float l = 0.0f;

    int i = tid;
    // 4 elements per iteration (i, i+128, i+256, i+384): coalesced loads
    // plus memory-level parallelism. Condition i+384 <= N-1 written as
    // (N - i) > 384 to avoid any overflow concern.
    while (N - i > 384) {
        const float v0 = xr[i];
        const float v1 = xr[i + 128];
        const float v2 = xr[i + 256];
        const float v3 = xr[i + 384];

        const float mn = fmaxf(fmaxf(fmaxf(m, v0), v1), fmaxf(v2, v3));
        const float s  = EXP(v0 - mn) + (EXP(v1 - mn) + (EXP(v2 - mn) + EXP(v3 - mn)));
        l = fmaf(l, EXP(m - mn), s);   // rescale old sum by exp(m - mn)
        m = mn;

        i += 512;                       // 128 threads * 4 elements
    }
    // Remainder: fewer than 4 strided elements left for this thread.
    for (; i < N; i += 128) {
        const float v  = xr[i];
        const float mn = fmaxf(m, v);
        l = fmaf(l, EXP(m - mn), EXP(v - mn));
        m = mn;
    }

    // ------------------------------------------------------------------
    // Block-wide reduction of (max, sum) across the 4 warps.
    // ------------------------------------------------------------------
    __shared__ float smax[4];
    __shared__ float ssum[4];

    const float m_own = m;              // this thread's own max (needed later)

    // warp-level max reduction (xor butterfly: all lanes get the result)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    if (lane == 0) smax[warp] = m;
    __syncthreads();

    // Global row max (every thread computes it from shared memory).
    const float M = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(smax[2], smax[3]));

    // Rescale this thread's partial sum to the global max, then reduce.
    // m_own - M <= 0 always; for idle threads l == 0, so the product is 0
    // regardless of the (possibly huge negative) exponent argument.
    float ls = l * EXP(m_own - M);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ls += __shfl_xor_sync(0xffffffffu, ls, off);

    if (lane == 0) ssum[warp] = ls;
    __syncthreads();

    const float L    = ssum[0] + (ssum[1] + (ssum[2] + ssum[3]));
    const float invL = 1.0f / L;        // L >= 1 (the max element contributes 1)

    // ------------------------------------------------------------------
    // Pass 2: y[i] = exp(x[i] - M) / L   (second, coalesced read of x)
    // ------------------------------------------------------------------
    i = tid;
    while (N - i > 384) {
        const float e0 = EXP(xr[i]       - M);
        const float e1 = EXP(xr[i + 128] - M);
        const float e2 = EXP(xr[i + 256] - M);
        const float e3 = EXP(xr[i + 384] - M);
        yr[i]           = e0 * invL;
        yr[i + 128]     = e1 * invL;
        yr[i + 256]     = e2 * invL;
        yr[i + 384]     = e3 * invL;
        i += 512;
    }
    for (; i < N; i += 128)
        yr[i] = EXP(xr[i] - M) * invL;
}
