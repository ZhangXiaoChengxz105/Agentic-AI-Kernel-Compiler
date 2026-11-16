// softmax_glm.cu — row-wise softmax, one CTA (128 threads) per row, NVIDIA A100 (sm_80).
// - Numerically stable: subtracts row max before EXP.
// - Works for any N >= 1 (handles ragged tails; no assumptions on divisibility).
// - Data-independent control flow: all branches depend only on threadIdx/N;
//   all value comparisons use fmaxf / shuffle reductions.
// - Fast path: row cached in static shared memory (single global read of x,
//   single global write of y) for N <= SMEM_FLOATS; global two-pass+scale
//   fallback for very long rows. Fully coalesced stride-128 accesses.
//
// Verifier-safety notes:
//   * The max-identity constant is -INFINITY (not -FLT_MAX / -3.4e38): VOLTA
//     represents float constants as exact rationals over i128, and any
//     constant with magnitude above ~1.7e38 overflows that representation.
//     -INFINITY is the natural max identity and is exactly representable to
//     the verifier. fmaxf(v, -INFINITY) == v for every finite/inf/nan-free
//     input, so control flow and results are unchanged.
//   * Max/sum reduction scratch buffers are disjoint, so each shared slot is
//     written exactly once per phase and only read after the barrier that
//     follows (no cross-phase races).

#include "common.cuh"

#define TPB 128          // blockDim.x, exactly as required
#define NWARP (TPB / 32) // 4 warps
#define SMEM_FLOATS 8192 // 32 KB static shared row cache (under 48 KB limit)

extern "C" __global__ void __launch_bounds__(TPB)
softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Row base pointers (64-bit offset math to be safe for large matrices).
    const size_t base = (size_t)blockIdx.x * (size_t)N;
    const float* __restrict__ xr = x + base;
    float* __restrict__ yr       = y + base;

    __shared__ float sred_max[NWARP];   // inter-warp max partials
    __shared__ float sred_sum[NWARP];   // inter-warp sum partials (disjoint!)
    __shared__ float sbcast[2];         // broadcast slots for block results
    __shared__ float sdata[SMEM_FLOATS]; // row cache for the fast path

    const bool use_smem = (N <= SMEM_FLOATS); // depends only on N: verifier-safe

    // ------------------------------------------------------------------
    // Pass 1: row max (and stage the row into shared memory if it fits).
    // ------------------------------------------------------------------
    const float NEGINF = -__int_as_float(0x7f800000); // -INFINITY: exact rational for VOLTA
    float m = NEGINF;
    if (use_smem) {
        for (int i = tid; i < N; i += TPB) {
            const float v = xr[i];   // coalesced, stride-1 across threads
            sdata[i] = v;
            m = fmaxf(m, v);
        }
    } else {
        for (int i = tid; i < N; i += TPB) {
            m = fmaxf(m, xr[i]);     // coalesced
        }
    }

    // Warp-level max reduction (data-independent loop bounds).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) sred_max[warp] = m;
    __syncthreads();

    // First warp reduces the NWARP partial maxima -> broadcast via sbcast[0].
    // sbcast[0] is written once here and never written again in the kernel.
    if (warp == 0) {
        float t = (lane < NWARP) ? sred_max[lane] : NEGINF;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, off));
        if (lane == 0) sbcast[0] = t;
    }
    __syncthreads();
    m = sbcast[0]; // safe: no thread writes sbcast[0] again before the next barrier

    // ------------------------------------------------------------------
    // Pass 2: exp(x - max) and its sum.
    //   Fast path:  exponentials live in shared memory (each thread later
    //               re-reads only slots it wrote itself -> no extra sync).
    //   Slow path:  unnormalized exponentials written straight to y (L2-hot
    //               for the final rescale pass).
    // Sum partials go to sred_sum (never sred_max / sbcast), so the reads of
    // the max above cannot race with these writes.
    // ------------------------------------------------------------------
    float s = 0.0f;
    if (use_smem) {
        for (int i = tid; i < N; i += TPB) {
            const float e = EXP(sdata[i] - m);
            sdata[i] = e;
            s += e;
        }
    } else {
        for (int i = tid; i < N; i += TPB) {
            const float e = EXP(xr[i] - m);
            yr[i] = e;
            s += e;
        }
    }

    // Warp-level sum reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) sred_sum[warp] = s;
    __syncthreads();

    // First warp reduces the NWARP sum partials -> broadcast via sbcast[1].
    if (warp == 0) {
        float t = (lane < NWARP) ? sred_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_down_sync(0xffffffffu, t, off);
        if (lane == 0) sbcast[1] = t; // second (and final) slot of sbcast
    }
    __syncthreads();
    const float inv = __frcp_rn(sbcast[1]); // reciprocal of the row sum

    // ------------------------------------------------------------------
    // Pass 3: write normalized output (coalesced).
    // ------------------------------------------------------------------
    if (use_smem) {
        for (int i = tid; i < N; i += TPB)
            yr[i] = sdata[i] * inv;
    } else {
        for (int i = tid; i < N; i += TPB)
            yr[i] = yr[i] * inv;
    }
}
