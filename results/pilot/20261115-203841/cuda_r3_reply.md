# cuda round 3

model=glm-5.3 in=3033 out=2764 cost=0.01575 latency=14.8s finish=stop effort=high usage={"completion_tokens": 2764, "completion_tokens_details": null, "prompt_tokens": 3033, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 1728, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 1459, "total_tokens": 5797}

## reasoning
The verifier is still failing with rational coefficient overflow. Even single exponentials of exact-real arguments like EXP(-200) presumably get represented as rationals... exp(-200) as a rational e^-200 is fine as a value but the verifier probably represents EXP as symbolic with rational coefficient arithmetic? Actually "rational coefficient overflowed i128" — likely the verifier models EXP(v) exactly (e^v as a transcendental, or as a rational approximation?) Hmm.

Wait — perhaps the issue is the initial value -FLT_MAX. EXP(x - M) where x = -FLT_MAX... Actually threads with no elements have m = -FLT_MAX, but that only enters pass 1, not pass 2 (l = 0). Hmm.

Actually let me think: maybe the verifier treats EXP(v) as e^v symbolically, and then expressions like sum of e^{x_i - M} over 256 elements with x_i as symbolic inputs... coefficients of e^{...}? Rational coefficient overflow i128 could come from summing many terms, or from FLT_MAX constants (3.4e38 doesn't fit i128? it does, i128 max ~1.7e38 — actually FLT_MAX = 3.4028e38 > 1.7014e38!). Yes! 2^127 ≈ 1.7e38, and FLT_MAX ≈ 3.4e38 which overflows i128 when the verifier tries to represent -FLT_MAX as an exact rational numerator!

That's the bug: -FLT_MAX = -2^128 * (2 - 2^-23) ≈ -3.4e38, exceeds i128 range (max 2^127 - 1 ≈ 1.7e38). Fix: use a smaller sentinel, e.g. -1e30f or -3.4e30f, or better: initialize m from the first element? But control flow must be data-independent — loading first element is fine (control flow doesn't depend on value). Actually simplest: initialize m = -3.0e30f (fits i128) or use INFINITY? INFINITY as rational is problematic too. Use -1e30f — fine since any real x > -1e30 presumably; but x could be -FLT_MAX... Well, x ~ 100 magnitude, and even if x = -1e38, fmaxf(m=-1e30, v=-1e38) = -1e30 which is wrong (max should be -1e38). Then EXP(v - m) = EXP(-1e38 - (-1e30)) = 0, all terms 0, L = 0, division by zero. Hmm.

Better: initialize m by loading xr[tid] if tid < N? But we need control flow depending on N only, which is allowed. Alternative: since exp underflow: if we use a sentinel m0 and the true max is below m0, then EXP(x_i - M) with M wrong... The result would be 0/0.

Alternative approach: make m initialized to xr[0]? Load first element: m = xr[0] (all threads read xr[0], broadcast). That's data-dependent value but not control flow. fmaxf(m, v) — correct max. That works for any N >= 1. If N >= 1, xr[0] exists. All threads load same address — fine.

But also could there be other overflow sources: 2^127 limit. Values of x themselves ~100 fine. Sums fine. FLT_MAX constant is the only thing exceeding i128. Also FLT_MAX appears only in initialization. So fix: m = xr[0].

But wait, what about pass 2 sum: fine. Also `-FLT_MAX` also implicitly via... only there. Also, could "rational coefficient overflow" come from e^x with x=-200? e^-200 ≈ 1.38e-87, representable as rational? As exact rational e^-200 is irrational; the verifier probably treats EXP as uninterpreted function symbol, coefficients are the linear combination coefficients — sums of e^{x_i} terms with coefficient 1. Those don't overflow. So the -FLT_MAX = -3.4e38 constant overflowing i128 is the most plausible cause.

Also check: does anything else use huge constants? FLT_MAX only. Also `1.0f/L` removed already.

Also note in pass 1, remaining threads with no elements: m = xr[0], harmless (a real data value, ≤ some finite value; it participates in fmaxf reduction but since every element ≥ ... hmm, actually if some thread has no elements and contributes m = xr[0] to the max reduction — xr[0] is a real element of the row, so the true max is ≥ xr[0], so including xr[0] doesn't change the max. Correct!

Edge: N ≥ 1 guaranteed. Good.

Also is -FLT_MAX problematic elsewhere? size_t computations fine.

Also potential issue: __shfl_xor_sync masks 0xffffffff with blockDim 128 full warps — fine.

Another potential verifier concern: the verifier does "exact real arithmetic" — maybe it models expf as exact exp. Products/quotients of sums of exps... coefficients of e^{x_i}? In pass 3 previously we had division by L (sum of exps) — y_i = e^{x_i-M} / L. If reference computes exp(x_i - max)/sum, symbolic equivalence: max computed by fmaxf chain vs reference max — equal. Sum association order differs but addition is associative-commutative in exact arithmetic — fine. So the only i128 overflow is the FLT_MAX constant.

So the fix: replace -FLT_MAX initialization with m = xr[0].

Also we might double-check: `while (N - i > 384)` — N-i could be... i starts at tid ≤127, N≥1, N-i could be negative (e.g., N=1, tid=127 → -126 > 384 false, ok). Negative fine.

Also common.cuh EXP macro — keep.

Also remove #include <cfloat> or keep (harmless). Keep but no longer needed; keep include is fine, but FLT_MAX no longer used. I'll drop cfloat include? It's fine either way; keep it minimal — remove it.

Wait, also should consider: maybe verifier also overflowed because of products like EXP(m-M) earlier — but we removed those and it still failed with identical message. So indeed the remaining suspect is -FLT_MAX.

Alternatively the verifier might represent -FLT_MAX as -340282346638528859811704183484516925440 which exceeds i128 max 170141183460469231731687303715884105727. Yes, overflow. Confirmed.

Provide explanation and file.

## content
```cuda
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
```
