# cuda round 1

model=glm-5.3 in=342 out=11469 cost=0.04918 latency=70.9s finish=stop effort=high usage={"completion_tokens": 11469, "completion_tokens_details": null, "prompt_tokens": 342, "prompt_tokens_details": null, "reasoning_tokens": 10175, "total_tokens": 11811}

## reasoning
Write a kernel with gridDim.x = rows, blockDim.x = 128. One row per block. Need online softmax (single pass) or two-pass using shared memory to store max and sum. Use warp shuffle reductions with mask... but mask depends on N, which is fine (control flow depends on N). Careful: __shfl_xor_sync with mask — mask should be constant 0xffffffff typically; using full mask is fine even if some threads inactive, as long as all threads in warp participate. If we loop over N with stride 128, all threads execute the loop; the loop condition depends on threadIdx and N only. Inside, we may access out-of-bounds; handle by loading conditionally — but conditional load `if (idx < N)` is control flow depending on threadIdx and N, that's allowed. Use max_val = fmaxf(max_val, (idx < N) ? x[idx] : -INFINITY) — but ternary is data-independent since idx and N are not data. Actually the ternary condition depends on threadIdx/N, allowed.

Alternative: use __ldg. Use vectorized loads? N arbitrary, so maybe not; keep simple but fast: coalesced 128-thread loads, two-pass online (single pass online softmax: track m and l). Online softmax single pass:

m_new = fmaxf(m, v); l = l * EXP(m - m_new) + EXP(v - m_new); m = m_new.

EXP(m - m_new) — careful when both are -INFINITY: m - m_new = -inf - -inf = NaN. Handle: initialize m = -INFINITY, l = 0. First element: m_new = v, l = l * EXP(m - m_new) = 0 * EXP(-inf)... 0 * EXP(-inf - v) — m - m_new = -inf - v = -inf (if v finite), EXP(-inf) = 0, 0*0 = 0. Fine. But if v is also -inf? x values are float32 arbitrary; assume finite? Safer: l = l * EXP(m - m_new) with m=-inf, m_new=-inf gives NaN. Guard: use fmaxf... Could avoid by processing: since control flow can't depend on data, we can't skip. Use a trick: l = l * EXP(m - m_new) + EXP(v - m_new); if both -inf: m-m_new = NaN. To be safe, initialize m = -3.4e38f? No, then m_new = fmaxf(m, v)... if v < m initially, fine; EXP(v - m_new) could overflow? v - m_new = v - (-3.4e38) large negative... Actually v - m where m = -3.4e38: if v is -3.4e38 too, difference = 0? Hmm.

Standard approach: initialize m = -INFINITY, and compute scale = (m_new == m) ? 1.0f : EXP(m - m_new)? That compares data. But `m_new == -INFINITY` data-dependent.

Alternative: two-pass. Pass 1 compute max, pass 2 compute sum, pass 3 write. That's 3 reads. Or pass 1 max, pass 2 sum + write (each element read once more). Online is single pass read but requires careful handling.

Actually the NaN concern: if x contains -inf. Problem says x ~ 100 large values; finite floats presumably. But "any N" and stable. I'll handle -inf safely anyway with a trick:

l = l * EXP(m - m_new): when m = -inf and m_new = -inf (first element also -inf), NaN. A common fix: use `float t = EXP(m - m_new); l = l * t + EXP(v - m_new);` and note that if m_new == m, t should be 1. We can compute t = EXP(m - m_new) but if m_new == m... can't branch on data.

Alternative fix: since l = 0 initially, l * NaN = NaN. Fix: l = fmaf(l, t, EXP(v - m_new))? NaN still.

Trick: replace -inf init with -FLT_MAX. Then m - m_new: if m_new = m = -FLT_MAX (all elements -FLT_MAX), difference = 0, EXP(0)=1, fine. If first element v, m_new = fmaxf(-FLT_MAX, v) = v (if v > -FLT_MAX). l = 0 * EXP(-FLT_MAX - v) = 0 * 0 = 0. Fine. If v = -FLT_MAX exactly, m_new = -FLT_MAX, EXP(v - m_new) = EXP(0) = 1, l = 1. Good. Overflow: m - m_new where m = -FLT_MAX and m_new = v finite: -FLT_MAX - 100 ≈ -3.4e38, EXP underflows to 0. Fine. What if v is +inf? EXP(v - m_new) = EXP(nan)? inf - inf = NaN. Edge case; ignore, or assume finite. Problem states float32 matrix, x ~ 100. I'll use -FLT_MAX init.

Also v - m_new could be large positive? No: m_new >= v always since m_new = fmaxf(m, v) >= v. So EXP(v - m_new) <= 1. Good.

Then after loop, block-reduce m and l across threads via warp shuffle + shared memory. But careful: online per-thread values must be merged: l_total = sum_i l_i * EXP(m_i - M). Merge via reduction where each thread contributes (m_i, l_i). Block reduction for max M first, then... but that requires two reductions and each thread needs l_i scaled. Since control flow can't depend on data, but reductions are fine (fmaxf, +).

Plan: 
1. Each thread computes local (m, l) online over its strided elements.
2. Block reduce max: warp shuffle reduce fmaxf, then shared, warp 0 reduces, broadcast M via shared.
3. Each thread computes l_scaled = l * EXP(m - M). If m = -FLT_MAX (thread had no elements, only possible if N < blockDim... N>=1, threads with idx >= N have no elements, m = -FLT_MAX, l = 0). EXP(-FLT_MAX - M): if M finite, underflow to 0, l*0=0 fine. If M = -FLT_MAX (all elements -FLT_MAX... M >= element values so M = -FLT_MAX only if all = -FLT_MAX): m - M = 0, EXP=1, l = 0. Fine. Actually could M - m overflow positive? m - M <= 0 always since M is max. Good.

Actually simpler: fold scaling into sum reduction: sum over threads of l_i * EXP(m_i - M). Use warp reduce with sum of scaled values.

4. Block reduce sum L.
5. Second pass over elements: y[idx] = EXP(x[idx] - M) / L. But threads that own elements re-read x — 2 reads total. Alternatively store per-element exp in... N could be large, no shared storage. Alternatively one-pass write: during online pass we can't write final values since M unknown. Could store values in registers if N/128 small — not general. So 3 reads per element total (pass1 read, plus re-read in pass2). Or two-pass: pass1 max only, pass2 compute sum, pass3 write: also multiple reads. Online approach: read x twice per element (once in online pass, once in write pass). That's good.

Alternative: keep per-thread scaled partial sums then write pass re-reads. Fine.

Shared memory: need for cross-warp reduction: 128 threads = 4 warps. Use two small arrays: float sdata[4]; and broadcast M and L. Also need M available in write pass: keep in shared or register after broadcast (register fine after sync). L likewise.

Loop bounds: control flow depends on threadIdx, blockDim, N — allowed. Use grid-stride... no, one block per row: for (int i = threadIdx.x; i < N; i += 128).

Write pass: for (int i = threadIdx.x; i < N; i += 128) y[i] = EXP(x[i] - M) * invL; where invL = 1.0f / L. Or divide; multiply faster. Compute oos = 1.0f / L... L >= EXP(v-M) summed >= something positive as long as at least one element; M = max, so at least one element has v == M, EXP(0)=1, L >= 1 (per owning thread l scaled: thread owning max element has l >= 1... actually l_i * EXP(m_i - M) >= 1 * 1 = 1). So L >= 1, safe.

Vectorization: could do float4 loads with boundary checks — complexity. For performance, maybe process 4 elements per thread per iteration when N >= 512? That complicates control flow but still N-dependent only. Given A100 and 128 threads/row, coalesced scalar loads achieve decent bandwidth. I could add float4 path: if (N >= 512) use vectorized loop... but the online update with 4 values: m_new = fmaxf(fmaxf(m, a), fmaxf(b, c))... fine. Boundary handling: tail loop scalar. Let me consider adding it — "aim for high performance". I'll implement: main vectorized loop over float4 while (i4 + 3 < N) handling via full warps... Actually simpler:

for (int i = threadIdx.x*4; i + 3 < N; i += 512) { load float4; online update with 4 vals; }
then scalar tail: for (int i = ... hmm need index mapping. Tail: elements from floor stuff. Let me do:

int i4 = threadIdx.x * 4;
vector loop: while (i4 + 3 < N) { ... i4 += 512; }
then scalar loop for remaining: indices i where i >= ? The remaining elements are those not covered: covered vector indices are [0, 4*floor(N/4)) roughly... Actually threads process i4 = tid*4 + k*512. All indices of form 4*tid + 512*k with +3 < N covered. The union of covered indices: all j such that j ≡ tid*4..tid*4+3 pattern... The full vectorized region covers indices [0, V) where V = N - (N mod 4)? Let's check: thread t covers blocks [4t + 512k, 4t+512k+3]. Since 512 = 128*4, thread t covers residues 4t,4t+1,4t+2,4t+3 mod 512, i.e., all residues mod 4 shifted... All threads together cover every 4-aligned block of 4. The loop condition i4+3 < N means block fully inside. So covered indices = [0, 4*floor(N/4)) — all of them? If N=10: floor(N/4)=2 blocks: [0..7]. Thread 0: i4=0: 3<10 yes covers 0-3; i4=512 no. Thread 1: i4=4: 7<10 yes covers 4-7. Remaining 8,9 in tail. Tail: start from j = 4*(N/4)? But which thread handles which tail element? Simplest: tail loop: for (int i = (N & ~3) + threadIdx.x; i < N; i += 128). N&~3 = 4*floor(N/4). Tail length < 4 ≤ 128, so each thread handles at most 1 tail element. Good, and control flow depends on N and threadIdx only.

Wait but if N < 4, tail covers everything. Good. If N=4: vector loop thread 0: i4=0, 3<4 yes. Good.

But careful: vector loop with while (i4 + 3 < N): different threads iterate different counts — fine, N/threadIdx dependent.

Also alignment: x + i4 where i4 multiple of 4 — but is x itself 16-byte aligned? The pointer passed as const float* __restrict__; row-major matrix, row r starts at x + r*N — NOT necessarily 16B aligned if N not multiple of 4! __ldg-style float4 load requires alignment. Row start alignment = alignment of x plus 4*N*r mod 16. If x is 16B aligned and N % 4 != 0, rows misaligned. Hmm. To be safe, either use __builtin_assume? Can't guarantee. Options: check alignment at runtime — that's data-independent (pointer value, not data values; control flow depending on pointer is probably fine since it's not data values... but verifier? "Control flow may depend only on threadIdx/blockIdx/blockDim/N" — pointer alignment is not in that list. Risky.)

Safer: scalar loads only. Or vectorize within guaranteed alignment: each thread loads 4 consecutive scalars — not vectorized hardware-wise but coalesced across threads anyway. Coalesced scalar loads on A100 achieve near peak bandwidth. Actually for memory-bound softmax, coalesced float loads are fine. Given verifier constraints, I'll do scalar loads with grid... loop stride 128. That's fully coalesced.

Hmm, but performance: could do float4 if N % 4 == 0 (branch on N — allowed!). N is a kernel parameter; control flow on N allowed. If N % 4 == 0, all rows are aligned iff x is 16B aligned. x is the matrix base; we can't guarantee 16B. Typically cudaMalloc gives 256B alignment. Could do: if (N % 4 == 0) vectorized else scalar — but still assumes base alignment. Many solutions assume it. To be robust, I could check ((uintptr_t)x & 15) == 0 — is that "data values"? It's pointer value, not data. The verifier likely instruments loads of data. Pointer arithmetic check is control flow on address, not on data. Hmm, ambiguous. The safest interpretation: control flow must not depend on values read from x/y. Pointer alignment check is fine. But "may depend only on threadIdx/blockIdx/blockDim/N" is strict. I'll avoid branching on pointer too; keep scalar loads. Simplicity + correctness over marginal speedup. Actually, let me reconsider: I could do vectorized load via __ldg on float4 only if N%4==0 AND aligned... I'll keep scalar to satisfy constraints strictly. Coalesced scalar loads: fine.

Actually, I can improve: each thread loads multiple elements per iteration with stride 128 (still coalesced): e.g., process 4 elements per iteration: i, i+128, i+256, i+384, then i += 512. This gives ILP without vector loads. Loop condition: i + 383 < N... plus tail handling. Complexity. Or simply: 

#pragma unroll 4 over strided loop — compiler handles; the loop trip count varies. Let me manually do unroll-by-4 with remainder handling:

int i = threadIdx.x;
// main: while (i + 384 <= N) { load 4, update, i += 512; }
Careful: condition i + 383 < N i.e. i+384 <= N. Elements covered: i, i+128, i+256, i+384. Then remainder loop for (int j = i; j < N; j += 128). Since we exited main loop when i+384 > N, remaining count per thread < 4. This covers all elements exactly once. Check: N=1: i=0 (thread0), main loop: 384<=1 false. Tail: j=0<1 handle. Good. N=130: thread 0: main: 384<=130 false; tail: j=0,128 → two elems. Thread 1: j=1,129. Good. N=1000: thread 0: i=0: 384<=1000 yes handle 0,128,256,384; i=512: 896<=1000 yes handle 512,640,768,896; i=1024: tail none (1024>=1000). Wait but 512+384=896 < 1000 yes. Elements 0..899 step 128 for thread 0... thread 0 covers {0,128,...,896}. Then i=512+512=1024 ≥ N. Tail loop none. But elements like 960? 960 = 128*7.5 → 960 mod 512 = 448 → thread (448 mod 128)... 960 = tid + 512k → tid = 960 - 512 = 448 > 127; 960 - 0 = 960 no. Hmm, 960/128 = 7.5, not integer. 960 mod 128 = 960 - 7*128 = 960-896 = 64. So tid=64, k = (960-64)/512 = 896/512 not integer. Hmm: elements are covered by tid = idx mod 128, iteration k = idx div 128. Thread 64: elements 64, 192, ..., 64+128k. 960 = 64 + 896 = 64 + 128*7, so k=7. In my unrolled scheme thread 64: i starts 64; main iteration handles i, i+128, i+256, i+384 with i = 64 + 512m. So elements 64+512m + 128c for c=0..3 → k = 4m + c, covering all k. i=64: covers k=0..3 (idx 64..448+64=512? i+384=448 → idx 448 = 64+384, k=3). Next i=576: covers idx 576..960 (k=4..7). Condition for i=576: 576+384=960 <= 1000? 960 <= 1000 yes → covers 576,704,832,960. Good, 960 covered. Next i=1088 ≥... 1088+384>N, tail: j from 1088: none. Thread 64 covered all its elements. 

But different threads exit main loop at different points and tail loops have varying trip counts — all dependent on N and threadIdx only. Good.

Hmm, but there's subtlety: thread t's main loop condition depends on i=t+512m. Coverage: each thread covers elements t+128k for k in [0, K_t). Main loop handles k in blocks of 4 while (t + 512m + 384) < N i.e. t + 512m + 384 <= N-1. The last element handled in block m is t+512m+384 = t+128(4m+3). So block m is fully processed iff t+128(4m+3) <= N-1, i.e., 4m+3 <= (N-1-t)/128. Elements beyond go to tail. Tail loop: for (int j = i; j < N; j += 128) where i = t + 512*(m_end). Covers exactly remaining. Correct.

This gives 4 outstanding loads per iteration → good ILP. Similarly for the write pass.

Actually simpler and still fast: just do the plain loop; compiler with #pragma unroll 4? Trip count unknown; compiler may unroll with remainder handling automatically. But explicit is safer for perf. I'll do explicit unroll-by-4 pattern. Or... let me weigh code complexity and correctness risk. I think I'll do unroll-by-4 carefully. Actually, maybe unroll 2 or 4. Let me write it.

Online update with 4 values:
float v0=..., v1, v2, v3;
m = fmaxf(fmaxf(fmaxf(m, v0), v1), fmaxf(v2, v3));
// then rescale l and add: careful order — need m_new first then l update:
Actually compute m_new = max of m and all four; then l = l * EXP(m - m_new) + EXP(v0-m_new)+EXP(v1-m_new)+EXP(v2-m_new)+EXP(v3-m_new). Since m - m_new <= 0 and v - m_new <= 0, safe. Use fmaf.

l = fmaf(l, EXP(m - m_new), EXP(v0-m_new)+...). Hmm EXP(m-m_new) when m = m_new (common) gives 1 → fine.

Edge: m init = -FLT_MAX. m - m_new = -FLT_MAX - v: could be -inf if v large positive? -3.4e38 - 100 → still finite (-3.4e38), EXP underflow → 0, l=0*0=0. OK. If v = -FLT_MAX: m_new = -FLT_MAX, m - m_new = 0. Fine.

Then block reductions:

__shared__ float sred[4]; // warp maxes
Warp reduce m: for (offset=16..1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
lane 0 writes sred[warpId]; __syncthreads();
All threads read 4 values? For 128 threads = 4 warps: each thread computes M = fmaxf of sred[0..3] (all read from shared). That avoids second sync + broadcast. M = fmaxf(fmaxf(sred[0],sred[1]),fmaxf(sred[2],sred[3])). Then l_scaled = l * EXP(m - M). __syncthreads() before reusing sred (or use separate array sred2[4] to avoid extra sync... need sync anyway before warp sum reduce since all warps must have read sred. Use two arrays: smax[4], ssum[4], one sync after writing smax, read, compute scaled, warp-reduce sum, write ssum[warp], sync, sum ssum[0..3]).

Wait: need syncthreads between writing smax and reading smax — yes. Then between reading smax (by all) and writing ssum — need another sync (same shared array reused) → use separate arrays to avoid: smax[4], ssum[4]. Then: sync after both writes? Sequence:

1. warp reduce max; lane0 smax[w] = m; 
2. warp reduce sum of l*EXP(m-M) — but M needs smax from all warps... so must sync first. OK:
   - warp max reduce, lane0 → smax[w]
   - __syncthreads()
   - M = reduce smax (all threads compute)
   - ls = l * EXP(m - M); warp sum reduce; lane0 → ssum[w]
   - __syncthreads()
   - L = ssum[0]+...+ssum[3]
3. inv = 1/L (or __frcp). Use 1.0f / L.
4. write pass: same loop structure; y[i] = EXP(x[i] - M) * inv.

Edge: l * EXP(m - M): m could be -FLT_MAX (thread with no elements, possible when N < 128). m - M: if M = -FLT_MAX? M >= all elements; M = -FLT_MAX only if all elements = -FLT_MAX (since M is max of thread maxes; a thread owning elements has m >= its elements ≥ -FLT_MAX; M = -FLT_MAX implies all owned elements = -FLT_MAX... actually fmaxf(-FLT_MAX, -FLT_MAX) = -FLT_MAX, yes possible). Then m - M = -FLT_MAX - (-FLT_MAX) = 0, EXP=1, l=0. Fine. If M finite and m=-FLT_MAX: m-M = -3.4e38 → EXP → 0, 0*0=0. Fine. Overflow of m - M positive? m <= M always? M is max over threads of m, so m <= M. m - M <= 0. Good. No NaN since no infs (assuming finite inputs; even -inf input: v=-inf: m_new = fmaxf(-FLT_MAX, -inf) = -FLT_MAX? fmaxf(-FLT_MAX, -inf): fmaxf returns the larger; -FLT_MAX > -inf, so m stays -FLT_MAX. v - m_new = -inf - (-FLT_MAX) = -inf → EXP = 0. OK actually -inf handled gracefully! +inf input: m_new = inf; then next m - m_new = inf - inf = NaN. Whatever, +inf input is out of scope.)

EXP macro from common.cuh — I must #include "common.cuh" and use EXP(v). I'll assume it's defined there (like #define EXP(v) __expf(v) or expf). Just use EXP.

Warp shuffles: __shfl_xor_sync(0xffffffff, ...) with full mask — all threads in warp active? Threads in warp all execute the reduction code since it's not inside data-dependent branch. Yes, the reduction is at top level after loops. All 128 threads reach it. Full mask fine.

But wait: warp reduce via shfl requires all lanes participate — they do.

Shared memory: static __shared__ float smax[4], ssum[4]. No dynamic shared. Good.

Also blockDim.x = 128 exactly, 4 warps. Hardcode 4? Use blockDim.x/32 — but requirement says exactly 128; I can hardcode constants but safer to compute from blockDim. Since blockDim is 128, warps = 4. I'll write generic-ish: const int nWarps = blockDim.x >> 5; but shared arrays sized... static arrays of 32 max (1024/32). Since blockDim fixed at 128, just use 4. I'll write with 4 hardcoded but compute warp id via threadIdx.x >> 5.

Let me also handle N < 1? N >= 1 given.

Write pass loop structure mirrors read pass. y[i] = EXP(x[i]-M)*invL. Note x re-read — use __ldg? x is const __restrict__ so compiler uses LDG anyway.

Now write code:

```cuda
#include "common.cuh"

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const float* __restrict__ xr = x + (size_t)row * (size_t)N;
    float* __restrict__ yr = y + (size_t)row * (size_t)N;

    // ---- Pass 1: online (single-read) max & sum of exp ----
    float m = -3.402823466e38f; // -FLT_MAX
    float l = 0.0f;

    int i = tid;
    // 4 elements per iteration for ILP
    while (i + 384 < N) {   // i, i+128, i+256, i+384 all valid
        float v0 = xr[i];
        float v1 = xr[i + 128];
        float v2 = xr[i + 256];
        float v3 = xr[i + 384];
        float mn = fmaxf(fmaxf(fmaxf(m, v0), v1), fmaxf(v2, v3));
        l = fmaf(l, EXP(m - mn), EXP(v0 - mn) + (EXP(v1 - mn) + (EXP(v2 - mn) + EXP(v3 - mn))));
        m = mn;
        i += 512;
    }
    for (; i < N; i += 128) {
        float v = xr[i];
        float mn = fmaxf(m, v);
        l = fmaf(l, EXP(m - mn), EXP(v - mn));
        m = mn;
    }
    ...
}
```

Wait condition: "i+384 < N" vs "<= N-1" — same thing. i+384 <= N-1 ⇔ i+384 < N. ✓.

Hmm wait, but there's an issue with the unrolled loop and per-thread coverage: thread t processes elements t+128k. In unrolled loop, block m covers k=4m..4m+3 requiring element t+128(4m+3) valid. Tail loop handles the rest. ✓ (verified above).

Warp reduce:

```cuda
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        l = l + __shfl_xor_sync(0xffffffffu, l, off);
    }
```

Wait — can I reduce both m and l simultaneously before scaling? No! l values from different threads have different local maxes; must scale by global M before summing. So reduce m across warps first, then scale. So:

```cuda
    // warp-level max reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    __shared__ float smax[4];
    __shared__ float ssum[4];

    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) smax[warp] = m;
    __syncthreads();

    float M = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(smax[2], smax[3]));
    // scale local sum to global max and reduce
    float ls = l * EXP(m - M);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ls += __shfl_xor_sync(0xffffffffu, ls, off);
    if (lane == 0) ssum[warp] = ls;
    __syncthreads();

    float L = ssum[0] + ssum[1] + ssum[2] + ssum[3];
    float invL = 1.0f / L;
```

Wait, all threads have same m after warp reduction (butterfly gives all lanes the result). Yes, xor butterfly: all lanes end with total. So l*EXP(m-M): each thread's own l and m... no wait — after the warp reduction of m, each thread's m register holds the warp max (same across warp), but l still holds its own local sum. That's fine: I need each thread's own (m_thread, l_thread). Oops! The butterfly overwrote m with warp max, destroying the thread's own m needed for scaling!

Fix: keep own m in a separate variable before reducing, or scale before reducing m. But scaling needs global M which needs cross-warp... Order:

1. warp-reduce m → each thread's m becomes warp max W_m (all lanes same). But I need original per-thread m for l scaling: l_own * EXP(m_own - M). Since after butterfly all lanes in warp have W_m, and m_own is lost.

Alternative: save own m: float mown = m; then reduce. Then ls = l * EXP(mown - M). Since all lanes have W_m but mown per-thread differs. Use separate variable. Fine.

Actually note: could scale l by EXP(mown - W_m) per warp and then warp-sum, then scale warp sums by EXP(W_m - M) — same thing. Simplest: save mown.

```cuda
    float mown = m;
    for (...) m = fmaxf(m, shfl(m));
    ...
    float ls = l * EXP(mown - M);
```

Check EXP(mown - M): mown could be -FLT_MAX for threads w/o elements → mown - M very negative → EXP → 0 → ls = 0*... wait l=0 anyway, 0 * EXP(negative huge) = 0 * 0 = 0. Fine. If M = -FLT_MAX: mown - M = 0, EXP=1, ls = l = 0. Fine.

Also potential issue: EXP(mown - M) where mown - M is exactly 0 → 1, good.

3'. Then final write pass:

```cuda
    i = tid;
    while (i + 384 < N) {
        float o0 = EXP(xr[i]       - M) * invL;
        float o1 = EXP(xr[i + 128] - M) * invL;
        float o2 = EXP(xr[i + 256] - M) * invL;
        float o3 = EXP(xr[i + 384] - M) * invL;
        yr[i] = o0; yr[i+128] = o1; yr[i+256] = o2; yr[i+384] = o3;
        i += 512;
    }
    for (; i < N; i += 128)
        yr[i] = EXP(xr[i] - M) * invL;
```

Should I add __syncthreads() before reusing... not reusing anything. Done.

One more concern: the tail `for (; i < N; i += 128)` — after unrolled loop, i for thread t is t + 512*m_end where m_end is number of full blocks. Tail covers remaining elements of this thread. But wait — is it guaranteed that all elements < N assigned to thread t are covered by main+tail? Main covers k=0..4*m_end-1 (elements t+128k for k < 4*m_end) where m_end = number of iterations. Tail starts at i = t + 512*m_end = t + 128*(4*m_end), continues while < N. Yes, covers exactly k = 4*m_end.... ✓. And main loop exit condition: after m_end iterations, i+384 >= N, i.e., t+128(4*m_end+3) >= N, so k=4*m_end+3 out of range — consistent.

Different threads different m_end — divergence at warp level for the tail loop, but tail is short (< 4 elements per thread, < 512 total... actually tail length per thread ≤ 3 iterations? k remaining < 4 per thread). Fine.

Hmm wait, tail per-thread iterations: after main loop, remaining k values for this thread: k in [4*m_end, ceil((N-t)/128)). Since 4*m_end+3 >= (N-1-t)/128... remaining ≤ ... Let me just trust it's ≤ 3 or so. Fine either way.

Loop condition `i + 384 < N` — signed int overflow? i ≤ N ≤ INT_MAX-ish; i+384 could overflow if N near INT_MAX. Assume N reasonable. Could use long long but perf. Use int; N as int parameter implies N ≤ INT_MAX. i+384 where i < N: if N close to INT_MAX, overflow UB. Use `i + 384 < N` → replace with `N - i > 384` to avoid overflow: N - i could be computed safely (N ≥ i? i < N guaranteed when in loop... entering loop first time i = tid < 128; if N ≤ tid, N - i negative, fine no overflow). `N - i > 384` avoids overflow. Similarly the loads i+384 etc are < N so within array. I'll use (N - i) > 384.

Hmm, `N - i > 384` means i+384 < N... wait N - i > 384 ⇔ i + 384 < N. But I want i+384 <= N-1 ⇔ i+384 < N. Yes same. But should it be >= 385? N - i > 384 ⇔ N > i + 384 ⇔ i + 384 ≤ N - 1. ✓ element i+384 valid (index ≤ N-1). ✓.

Now the "common.cuh" include — I just include it and use EXP. Also may need math constants; FLT_MAX needs <cfloat> or <float.h>. I'll include <cfloat>. Actually common.cuh probably includes stuff, but include <cfloat> to be safe. Also fmaf needs <cmath> — usually available via cuda runtime headers. Include <cmath> too. Hmm, "no external libraries" — standard headers fine? cuda's common.cuh presumably defines EXP. I'll include <cfloat> and rely on fmaxf/fmaf built-ins (CUDA device built-ins, declared in device code via crt). Safe.

Define -FLT_MAX literal: -3.40282346638528859811704183484516925e38f. Use -FLT_MAX with <cfloat>.

One more consideration: `if (lane == 0) smax[warp] = m;` — control flow depends on threadIdx — allowed.

Requirement: blockDim.x = 128 exactly, grid = rows. The kernel doesn't control launch but should be written for that. Hardcode 4 warps? I'll use blockDim-based but with static shared arrays sized 4 (since blockDim fixed 128). Fine — write `const int warp = threadIdx.x >> 5;` and arrays [4]. If launched with 128 as specified, correct.

Also might use __ldg explicitly? const __restrict__ suffices.

Let me also double check the online update numerical stability with x ~ ±100: m - mn ∈ [-200, 0], EXP ≤ 1. v - mn ≤ 0, EXP ≤ 1. l ≤ N. All good. Final EXP(x-M) ≤ 1, sum L ∈ [1, N]. No overflow.

Potential precision: fmaf(l, EXP(m-mn), sum) — fine.

Also `EXP(m - mnew)` when m == mnew → EXP(0) = exp(0) = 1 exactly (even __expf(0)=1). Good.

Edge case N=1: only thread 0 has element. m of thread 0 = x[0]; others -FLT_MAX, l=0. M = x[0]. Thread0 ls = 1*EXP(0)=1. L=1. y[0] = EXP(0)*1 = 1. ✓.

Now also think: should the two __syncthreads be issues if some threads exited early? No early returns. All threads reach syncs. ✓.

Consider making the unroll 4 → maybe use 4 offsets {0,128,256,384} — yes.

Also, using size_t for row offset: (long long)row * N could overflow int for big matrices; use size_t cast. ✓.

Write final code with comments. Also add `#include "common.cuh"` first.

One thing: verifier "control flow may depend only on threadIdx/blockIdx/blockDim/N" — my loops: `while ((N - i) > 384)` depends on N and i (derived from threadIdx). ✓. `for (; i < N; i += 128)` ✓. `if (lane == 0)` ✓. Warp reduce loops: constant. ✓. No data-dependent branches. fmaxf used. ✓.

Ternaries? None on data. EXP macro only for exponentials — I call EXP only, never expf. ✓ (fmaf, fmaxf fine).

Also careful: `l = fmaf(l, EXP(m - mn), e0+e1+e2+e3)` — writes.

Final code:

```cuda
#include "common.cuh"
#include <cfloat>

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int   tid = threadIdx.x;
    const int   warp = tid >> 5;
    const int   lane = tid & 31;
    const float* __restrict__ xr = x + (size_t)blockIdx.x * (size_t)N;
    float* __restrict__ yr = y + (size_t)blockIdx.x * (size_t)N;

    // ---------- Pass 1: streaming online softmax (one read of x) ----------
    float m = -FLT_MAX;   // running max of this thread's elements
    float l = 0.0f;       // running sum of exp(v - m)

    int i = tid;
    // 4 elements per iteration (i, i+128, i+256, i+384) for memory-level parallelism
    while (N - i > 384) {
        const float v0 = xr[i];
        const float v1 = xr[i + 128];
        const float v2 = xr[i + 256];
        const float v3 = xr[i + 384];
        const float mn = fmaxf(fmaxf(fmaxf(m, v0), v1), fmaxf(v2, v3));
        const float s = EXP(v0 - mn) + (EXP(v1 - mn) + (EXP(v2 - mn) + EXP(v3 - mn)));
        l = fmaf(l, EXP(m - mn), s);
        m = mn;
        i += 512;
    }
    // remainder (fewer than 4 strided elements per thread)
    for (; i < N; i += 128) {
        const float v = xr[i];
        const float mn = fmaxf(m, v);
        l = fmaf(l, EXP(m - mn), EXP(v - mn));
        m = mn;
    }

    // ---------- Block-wide reduction (128 threads = 4 warps) ----------
    __shared__ float smax[4];
    __shared__ float ssum[4];

    // warp-level max reduction (butterfly)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    const float m_own = ... 
```

Wait — I need m_own saved BEFORE the warp reduction. Fix order:

```cuda
    const float m_own = m;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) smax[warp] = m;
    __syncthreads();

    const float M = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(smax[2], smax[3]));

    float ls = l * EXP(m_own - M);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ls += __shfl_xor_sync(0xffffffffu, ls, off);
    if (lane == 0) ssum[warp] = ls;
    __syncthreads();

    const float L = ssum[0] + (ssum[1] + (ssum[2] + ssum[3]));
    const float invL = 1.0f / L;

    // ---------- Pass 2: normalize and store ----------
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
```

Check `N - i > 384` when i ≥ N (e.g., N=1, tid=5): N-i = -4, not > 384, skip. Tail loop: i≥N skip. ✓. When i could exceed N such that N-i very negative — no overflow (both ints, |diff| < 2^31 fine as long as no overflow: N ≤ 2^31-1, i < N + 512ish... i max = tid + 512*k; k bounded by N/512 roughly; i < N + 512. N - i > -512. Fine).

Edge: what if N - i > 384 but some of i+128 etc exceed... no: i+384 ≤ N-1 guaranteed. ✓.

Double-check warp shuffle mask: all 32 lanes of every warp active at reduction points (no divergence there — the loops above have uniform-exit per thread but all threads eventually reach reduction). Within a warp, threads may exit the while loop at different iterations but they all converge before the reduction code (no sync needed for correctness of shuffle since all lanes execute the shuffle instructions). ✓.

Hmm, one subtle: divergent warps reaching __shfl_xor_sync with full mask — all threads eventually execute it; the sync variants require all threads in mask to execute the same instruction; since reconvergence happens at the loop exit (compiler inserts reconvergence), fine. Standard pattern.

Also m after reduction: only lane0's value written; other lanes' m unused. Fine.

`smax`/`ssum` shared arrays: 4 floats each, static shared — allowed (no dynamic shared memory).

Could also consider using __expf via EXP macro — whatever common.cuh defines; we just call EXP.

I think code is complete. Add a comment header. Also note: no use of `rows` — grid covers rows via blockIdx.x. Good.

One more perf thought: maybe use `#pragma unroll` hints on while loops — can't fully unroll unknown trip count; leave.

Also for pass 2, could reload from x — yes we do (needed since values not stored).

Potential issue: register pressure fine.

Deliver single code block.

## content
```cuda
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
```
