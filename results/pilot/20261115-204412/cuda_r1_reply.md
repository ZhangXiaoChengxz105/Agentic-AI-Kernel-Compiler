# cuda round 1

model=glm-5.3 in=342 out=2078 cost=0.00928 latency=12.2s finish=stop effort=high usage={"completion_tokens": 2078, "completion_tokens_details": null, "prompt_tokens": 342, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 320, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 1459, "total_tokens": 2420}

## reasoning
Requirements: one CTA per row, blockDim.x=128, handle any N. Use warp/block reductions with shuffles, no data-dependent control flow (loop bounds depend only on N and thread index — fine). Use EXP macro from common.cuh.

Two-pass over the row (max then sum) plus write pass, or single pass storing exp values in registers? N unknown size, so use global memory two passes: compute max, then sum of exp, then write y = exp(x - max)/sum. That's 3 reads of x actually (max pass, sum pass, write pass). Could store exp(x-max) in y during sum pass, then rescale in write pass — that avoids third read of x: pass 1 read x compute max; pass 2 read x, compute e = EXP(x-max), write e to y, accumulate sum; pass 3 read y, multiply by 1/sum. That's fine and coalesced.

Grid-stride style loop over N with stride 128.

Block reduction: warp shuffle down then shared memory across 4 warps.

Careful with data-independent control flow: reduction with shfl_xor is fine. Shared memory reduction: each warp lane 0 writes to s[warp], then first warp reduces; need __syncthreads. Use fixed loop for 4 warps (constexpr since blockDim=128 → 4 warps). Fine.

Handling max init: use -INFINITY, fmaxf fine. For sum, fine.

EXP of x - max where x = max gives exp(0)=1, sum >= 1 always, no divide by zero.

Write kernel:

```cuda
#include "common.cuh"

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* row = x + (size_t)blockIdx.x * N;
    float* orow = y + (size_t)blockIdx.x * N;

    // Pass 1: max
    float m = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x)
        m = fmaxf(m, row[i]);

    // warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    __shared__ float sred[4];
    if (lane == 0) sred[warp] = m;
    __syncthreads();
    m = sred[lane & 3]; // hmm, first warp reads 4 values
```

Simpler: after warp reduce, lane0 writes sred[warp]; syncthreads; then all threads read sred[0..3]? Let thread 0..3 reduce? To keep all threads with the final value: each thread computes m = sred[0..3] combined — all threads read 4 shared values and fmaxf them. Cheap.

```cuda
    __syncthreads(); // before reuse (need sync before writing sred again)
```

Actually reuse sred for sum: after reading, __syncthreads() before pass 2 writes to sred.

Pass 2:
```cuda
    float s = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        float e = EXP(row[i] - m);
        orow[i] = e;
        s += e;
    }
    warp reduce sum with shfl_xor add.
    if (lane==0) sred[warp] = s;
    __syncthreads();
    s = sred[0]+sred[1]+sred[2]+sred[3];
    float inv = 1.f / s;
    for (int i = tid; i < N; i += blockDim.x)
        orow[i] *= inv;
}
```

Wait: in pass-1 shared reduction, all threads read sred[0..3] — need syncthreads after write before read; yes. Then before pass-2 writes sred again, need syncthreads after all threads finish reading — the read happens right after first sync; add second sync before lane0 writes in pass 2. Actually pass 2's write to sred happens after the loop with computation, but a thread could still... warp execution: thread reads sred (after sync1), then does pass 2 loop, then lane 0 writes sred[warp]. Another warp's thread might still be reading sred from pass 1? No — sync1 ensures all writes done; reads after sync1 are safe. Writes in pass 2: need all reads of sred (pass1) done before overwrite. Reads happen immediately after sync1 but there's no sync between read and pass-2 write. Warp divergence could let one warp finish pass2 and write sred while another warp is... no, another warp must have executed past sync1 and its reads... not necessarily: reads after sync1 in warp A may occur later than pass-2 write by warp B if warp A is stalled. So need a sync between reading sred values and the next write. Add __syncthreads() after computing m from sred. Safer: just use two separate shared arrays sred[4] for max and ssum[4]. Still need sync before writing ssum? Writing ssum is first write to it, no prior reads. Fine — but keep syncthreads anyway for clarity. Actually with separate arrays, after sync1 (max written), all threads read smax, that's safe. Pass2 writes ssum — different array, no hazard. Then sync before reading ssum. But reuse across... it's a single kernel, one iteration, no reuse. Good.

Also for very small N: threads with no elements have m = -inf... if N>=1, at least thread 0 has value. fmaxf(-inf, -inf) = -inf fine. Threads with no data contribute s=0. Fine.

Edge: reading sred[0..3] — all 4 warps write sred even if some warps had no data (they still write -inf/0). Good since all threads execute lane==0 write.

Data-independent control flow: loops bound by i < N with i starting at tid — depends only on threadIdx and N. shfl_xor fine. All good.

Use __ldg? __restrict__ const should be enough. Maybe use vectorized loads? N not multiple — complicates. Keep simple scalar; coalesced.

Grid = rows: kernel doesn't need to check anything.

Let me write final code with unrolled loop.

EXP macro comes from common.cuh — assume EXP(v).

## content
```cuda
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
```
