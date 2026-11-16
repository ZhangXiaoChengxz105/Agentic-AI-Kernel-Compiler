# cuda round 2

model=glm-5.3 in=1885 out=2783 cost=0.01431 latency=13.7s finish=stop effort=high usage={"completion_tokens": 2783, "completion_tokens_details": null, "prompt_tokens": 1885, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 320, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 1207, "total_tokens": 4668}

## reasoning
The bug: after the second inter-warp reduction, thread t0 (warp 0 lane 0) writes sred[0], then reads `m = sred[0]`... wait, actually the race: after the second reduction (`if (lane == 0) sred[0] = t;`), we do `__syncthreads(); const float inv = ...`. That's fine.

Wait — the race is: write by t0 at pc:348 conflicts with read by t32 at pc:192. Likely: in the second warp reduction... let me trace.

Actually the issue: the final `__syncthreads()` before reading sred[0] — I did include `__syncthreads()` after `if (warp==0){...}`. Hmm, check my code:

```
    if (warp == 0) {
        ...
        if (lane == 0) sred[0] = t;
    }
    __syncthreads();
    const float inv = __frcp_rn(sred[0]);
```

That looks fine. But wait — the race is write t0 pc:348 vs read t32 pc:192. pc:192 is likely `m = sred[0]` read after the FIRST reduction... Actually think: first reduction: warp partials written to sred[warp], sync, warp 0 reduces, writes sred[0], sync, all read sred[0]. That's fine.

Hmm, but the issue: in the first reduction, threads of warp 0 read `sred[lane]` for lane < NWARP — that read happens after the first __syncthreads. OK.

But wait — there's a subtle race: after the FIRST reduction's final `__syncthreads(); m = sred[0];` — all threads read sred[0]. Then threads proceed to pass 2. In pass 2, `if (lane==0) sred[warp] = s;` writes sred[0] (warp 0). Is there a sync between the read of sred[0] (m = sred[0]) by all threads and the write sred[0] = s by t0? Between them: pass 2 loop, warp shuffle reduction... no __syncthreads! Actually the code:

```
    __syncthreads();
    m = sred[0];

    // pass 2
    ... loop ...
    // warp reduction
    if (lane == 0) sred[warp] = s;   // t0 writes sred[0]
    __syncthreads();
```

Between `m = sred[0]` (read by t32) and `sred[0] = s` (write by t0) there's no __syncthreads. Wait — the read is `m = sred[0]` after a __syncthreads, then the write happens later. Race: t0 might reach the write before t32 does the read. Yes! That's the race: read t32 pc:192 (m = sred[0]) vs write t0 pc:348 (sred[0]=s in second pass).

Fix: copy m into registers immediately... but the read itself is the problem: t32 reads sred[0] while t0 may already write it. Wait, but t32 reads it after __syncthreads which ensures t0's write from warp-0 reduction completed. But then t0 proceeds into pass 2 and writes sred[0] again before t32 has read it. Because __syncthreads only synchronizes at that point; after it, t0 races ahead.

Fix: add a __syncthreads after reading m = sred[0], or better: broadcast m via __shfl from warp 0? Simplest: read m into a register, then __syncthreads() before writing sred in pass 2. Or avoid sred[0] reuse: use separate shared variables sred for max and sred2 for sum. But still, the read of sred[0] and subsequent writes to sred (warp slots) — warp 0 writes sred[0] again. Use a separate array sred2 for the sum reduction. Then sred[0] is only written once (during max phase) and never again. That fixes it cleanly without extra sync.

Also same potential race at the end: `inv` reads sred2[0], no further writes. Fine.

But careful: sred[0] is read as `m = sred[0]` by all threads; the only writes to sred happen before that (during max reduction). After that, pass 2 uses sred2. No race.

Also the initial `if (lane == 0) sred[warp] = m;` before first sync — fine.

So fix: add second shared array `__shared__ float sred2[TPB];` for sum. Or just add __syncthreads after `m = sred[0]`. Both work; separate arrays is cleaner and race-free. Also do the same for the final read: `inv = sred2[0]` — writes to sred2 happen before the sync preceding. Fine.

Also the smem note: adding 128 floats = 512 bytes more. SMEM_FLOATS 8192 floats = 32KB; total 32KB + 1KB fine.

Alternatively simplest fix per hint: add __syncthreads() after reading m. I'll do both: use separate array (cleanest) — and it's also faster (no extra barrier).

Write full file.

## content
```cuda
// softmax_glm.cu — row-wise softmax, one CTA (128 threads) per row, NVIDIA A100 (sm_80).
// - Numerically stable: subtracts row max before EXP.
// - Works for any N >= 1 (handles ragged tails; no assumptions on divisibility).
// - Data-independent control flow: all branches depend only on threadIdx/N;
//   all value comparisons use fmaxf / shuffle reductions.
// - Fast path: row cached in static shared memory (single global read of x,
//   single global write of y) for N <= SMEM_FLOATS; global two-pass+scale
//   fallback for very long rows. Fully coalesced stride-128 accesses.
//
// Race fix vs previous revision: the max-reduction scratch (sred_max) and the
// sum-reduction scratch (sred_sum) are now SEPARATE arrays. Previously the sum
// phase overwrote sred[0] while slower threads were still reading the row max
// from it (write by t0 @ sum-phase vs read by t32 of `m = sred[0]`), because
// no barrier separated that read from the next write. With disjoint scratch
// buffers, each shared location is written exactly once per reduction phase
// and only read after the barrier that follows, so no cross-phase aliasing
// race exists and no extra __syncthreads() is needed.

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
    __shared__ float sbcast[1];         // broadcast slot for block results
    __shared__ float sdata[SMEM_FLOATS]; // row cache for the fast path

    const bool use_smem = (N <= SMEM_FLOATS); // depends only on N: verifier-safe

    // ------------------------------------------------------------------
    // Pass 1: row max (and stage the row into shared memory if it fits).
    // ------------------------------------------------------------------
    const float NEGINF = -3.402823466e38f; // -FLT_MAX
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

    // First warp reduces the NWARP sum partials -> broadcast via sbcast_sum[0].
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
```
