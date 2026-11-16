# cuda round 1

model=glm-5.3 in=342 out=24000 cost=0.10241 latency=495.4s finish=length

## reasoning
Let me think carefully about the requirements:

1. One CUDA C++ kernel for A100 (sm_80)
2. Row-wise softmax over row-major float32 matrix x[rows, N], output y[rows, N]
3. Exact signature: `extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)`
4. Launch: grid = rows (one CTA per row), blockDim.x = 128 exactly
5. Correct for ANY N >= 1 (N often not multiple of 128), numerically stable for large |x|
6. Every exponential as EXP(v), a macro from "common.cuh", must #include "common.cuh"
7. Control flow may depend only on threadIdx/blockIdx/blockDim/N, never on data values — use fmaxf(a,b) not if(a>b)
8. No dynamic shared memory, no cp.async, no cooperative groups, no external libraries
9. High performance: fast reductions, coalesced access

Key design:

- One CTA (128 threads) per row.
- Two-pass (or three-pass) online approach: 
  - Pass 1: compute row max via reduction.
  - Pass 2: compute sum of exp(x - max) via reduction.
  - Pass 3: write out y = exp(x-max)/sum.

Since control flow must be data-independent, we can't do the online single-pass (flash-style) rescaling? Actually the online softmax rescaling uses max updates — but with data-independent control flow we can still do the online algorithm since it doesn't need branches on data: new_max = fmaxf(old_max, m), correction = EXP(old_max - new_max), sum = sum * correction + EXP(x - new_max). That's branchless. But it's more EXP calls. Actually the online approach would compute EXP for each element once in pass computing sum, and then need to recompute EXP in the write-out pass (since we can't store all exps unless we use registers/local memory... we could cache values in registers if each thread handles a fixed number of elements, but N is arbitrary).

Simplest robust approach: 
- Pass 1 (max): each thread strides over the row with stride 128, computing local max (init to -inf), handling bounds via predicated loads (threadIdx.x + k*128 < N). Use -INFINITY as init; fmaxf handles it.
- Block reduce max via warp shuffles + shared memory.
- Pass 2 (sum): each thread computes local sum of EXP(x - max), with predication; out-of-bounds contribute 0.0f. Actually careful: if we compute EXP only for valid elements, we need control flow depending on threadIdx and N only — that's fine (index < N depends on threadIdx/N, not data).
- Block reduce sum.
- Pass 3 (write): y = EXP(x - max) / sum. Recompute EXP.

That's 2 reads of x (pass 1 and 2) plus 1 more read (pass 3) = 3 reads of the row, 1 write. Alternative: cache x in registers? N arbitrary, so each thread has variable count — can't do static arrays without a cap. Could store x in shared memory: N*4 bytes per row; with 48KB static shared memory, that's up to 12288 floats. But N can be anything; we'd need dynamic shared memory which is disallowed. Static shared memory fixed size limits N. Not good for "ANY N >= 1". So recompute EXP in pass 3: total 3 passes over data. Alternatively, store exp values... no.

Alternatively two passes: pass 1 computes max, pass 2 computes sum and we could compute exps and... we still need to write y. We could compute y in pass 2 if sum were known — it's not. So 3 passes: read x three times. For large N that's memory bound anyway; exp is fast enough. Actually reading 3x vs 2x: the two-pass approach with caching in shared memory would be 1 read + 1 write, but limited N. The 3-pass approach: 3 reads + 1 write = 4N*4 bytes traffic. The online approach: 2 reads + 1 write (max computed online along with sum in one pass, then a second pass to write) = 3N*4 bytes. Hmm, online softmax: pass 1 computes running max and running sum (with rescaling), pass 2 writes output using final max and sum, recomputing EXP. That's 2 reads + 1 write. Slightly better traffic. But per element it does 2 EXP calls in pass 1 (one for correction factor? no—correction is scalar per chunk) Let me think: online per element: EXP(x_i - new_max) once, plus scalar corrections per iteration. Pass 2: EXP(x_i - final_max) once. Total 2 EXP per element. The 3-pass approach: pass 1 no EXP, pass 2 one EXP, pass 3 one EXP → also 2 EXP per element. Same EXP count, but online has less memory traffic. However online is more complex. Given memory-bound nature, 2 reads vs 3 reads matters (33% less traffic). But wait — with grid = rows and 128 threads per row, for small N the GPU is underutilized regardless; for large N, memory bound; the 3-pass reads 3x. L2 might cache the row between passes if row fits in L2 (A100 has 40MB L2). Rows are contiguous; between pass 1 and pass 2, the same CTA re-reads the same row — likely still in L2 if row < some MB. So the extra pass may hit L2. Still, online approach is cleaner performance-wise but more code.

Hmm, but there's a subtlety with the online approach and data-independent control flow: none needed; all branchless.

Actually, let me reconsider: simpler is 3-pass. But let me weigh: the "aim for high performance" requirement. A common efficient pattern for softmax with one block per row:

Option A (3-pass, simple):
```
float m = -INF;
for (int i = tid; i < N; i += 128) m = fmaxf(m, x[i]);
m = block_reduce_max(m);
float s = 0;
for (int i = tid; i < N; i += 128) s += EXP(x[i] - m);
s = block_reduce_sum(s);
float inv = 1.0f / s;
for (int i = tid; i < N; i += 128) y[i] = EXP(x[i] - m) * inv;
```
Memory traffic: 3 reads + 1 write. 

Option B (online, 2-pass):
```
float m = -INF, s = 0;
for (int i = tid; i < N; i += 128) {
    float v = x[i];
    float nm = fmaxf(m, v);
    s = s * EXP(m - nm) + EXP(v - nm);
    m = nm;
}
// reduce: need to combine (m, s) pairs across threads: two-phase max then rescale
m = block_reduce_max(m);
s = s * EXP(m_local - m); // wait, each thread's s is relative to its own m
s = block_reduce_sum(s);
float inv = 1/s;
for (int i = tid; i < N; i += 128) y[i] = EXP(x[i] - m) * inv;
```
Wait, in the combine step: each thread's local sum s_t is sum of EXP(x - m_t). To combine, scale by EXP(m_t - m_global): s_t * EXP(m_t - m_global). Note EXP(m_t - m_global) where m_t could be -INF (thread with no elements or all -INF). EXP(-INF - m) = 0. But if m_global is also -INF (all elements -INF, i.e., N... N>=1 so there's at least one element; but that element could be -INF if input contains -INF; edge case: x values -INF. Typically inputs are finite. But "numerically stable for large |x|" — finite large. If m_t = -INF and m_global = -INF: EXP(-INF - (-INF)) = EXP(NaN) = NaN. Hmm. -INF - -INF = NaN. To be safe, guard: if m_t is -inf... that's data-dependent control flow though. Alternative: use fmaxf to avoid NaN? We could clamp: contribution = (m_t == -INFINITY) ? 0.f : s_t * EXP(m_t - m). The ternary on data... "Control flow may depend only on threadIdx/blockIdx/blockDim/N, never on data values (use fmaxf(a, b), not if (a > b))". A select via ternary is technically a select instruction, not control flow — but the verifier might flag it. Safer: avoid the -INF edge case by initializing m to -FLT_MAX or using a large negative finite number? If we init m = -3.4e38f, then for finite inputs all fine. If input contains actual -INFINITY, then v = -INF, nm = fmaxf(-3.4e38, -INF) = -3.4e38 (fmaxf returns the non-NaN max; -INF < -3.4e38 so nm = -3.4e38). Then EXP(v - nm) = EXP(-INF) = 0. Fine. If input contains NaN — fmaxf(NaN, m) returns m (fmaxf ignores NaN), so NaN inputs get treated... y would be EXP(NaN - m) = NaN in output. Whatever, NaN in → NaN out is acceptable.

Also with 3-pass approach and init -INF: pass 2 EXP(x - m) where if N>=1 there's at least one finite-ish max; if the max is -INF (all inputs -INF), then sum = sum EXP(-INF - -INF)= NaN. Edge case unlikely to be tested. Using -FLT_MAX init: if all inputs are -INF, m = -FLT_MAX, EXP(-INF + FLT_MAX) = EXP(-inf)=0, sum=0, inv = inf, y = 0 * inf = NaN. Any approach breaks on all -INF row. Standard. I'll init with -FLT_MAX or -INFINITY? Let me think about which is safer for typical verifier tests: inputs are finite floats. Both work. I'll use -INFINITY actually... hmm, but in option A pass 2: EXP(x[i] - m) with m = -INFINITY only when all elements are -INFINITY, giving NaN — but x[i] = -INFINITY too, EXP(-INF - -INF) = EXP(NaN) = NaN. With -FLT_MAX init: m=-FLT_MAX, EXP(-INF + 3.4e38) = EXP(-INF) = 0. Sum 0. y = 0/0 → NaN * ... hmm both NaN. Fine, all-(-INF) rows are degenerate; I'll not worry, but I'll use -FLT_MAX... Actually wait: what about using -INFINITY and the row being fine — no issue. Let me just use -INFINITY for max init; standard. Hmm, but there's one more subtle case: threads with no elements (e.g., N < 128, threads with tid >= N). Their local max = -INFINITY, contribute to block max reduction fine (fmaxf ignores). In block sum reduction they contribute 0. Fine.

Let me reconsider Option B's thread-local combine more carefully. After the loop, each thread has (m_t, s_t). Block reduce max over m_t → M. Then each thread computes s_t * EXP(m_t - M) and block-reduce sum → S. For threads with no elements: m_t = -INF, s_t = 0. EXP(-INF - M): if M finite, = 0, times s_t=0 → 0. Good. If M = -INF (all -INF row, degenerate): NaN. Degenerate anyway.

But wait — if I init m_t = -FLT_MAX instead: threads with no elements have m_t = -FLT_MAX, s_t = 0 → 0 * EXP(-FLT_MAX - M) = 0 * 0 = 0. Fine. All-(-INF) row: M = -FLT_MAX, s_t = 0, S = 0, y = EXP(-INF + FLT_MAX)*inv = 0 * inf = NaN. Degenerate. OK.

However, there's a numerical concern with Option B: EXP(m_t - M) — if m_t is very negative and M large, EXP underflows to 0, correct behavior. And s_t * EXP(m_t - M): s_t could be up to N per thread... fine, no overflow since s_t ≤ count and EXP ≤ 1.

Also per-iteration correction: s = s * EXP(m - nm) + EXP(v - nm). EXP(m - nm) where m=-FLT_MAX initially, nm = v: EXP(-huge) = 0. Good, s starts 0 anyway.

Actually hold on, one more subtlety in Option B: within the loop, EXP(m - nm) when m == nm gives EXP(0)=1, fine.

Option B does 2 EXP per element in pass 1 (one for v, one for the scalar correction per element — correction is per iteration, i.e., per element). So per element: 2 EXPs in pass 1 + 1 EXP in pass 2 = 3 EXPs per element. Option A: 0 + 1 + 1 = 2 EXPs per element. EXP on A100 (MUFU.EX2 path via expf → __expf? EXP macro — unknown implementation, likely expf or __expf). If EXP is expf (accurate), it's a slow software-ish function? No — expf in CUDA compiles to a sequence: roughly fma + ex2.approx, about 4-6 instructions. __expf is ex2.approx.ftz plus a multiply. Either way it's cheap-ish arithmetic. The kernel is memory bound for large N; for small N (row per CTA, many rows), it's launch/occupancy bound. EXP count matters less than memory traffic for big N. But for medium N that fits in L1/L2... hmm.

Actually, let me think about what's really the performance driver here. Grid = rows, 128 threads/CTA. 

Case 1: N small (e.g., N ≤ 128 or up to a few thousand) and rows large → many CTAs, each does little work. Performance bound by kernel launch throughput and reduction overhead. Here, number of passes over global memory matters less (row is in L2/L1 after first touch); the block reductions (syncthreads count) matter.

Case 2: N large (e.g., N = 100k), rows moderate. Memory bound. 3 reads + 1 write vs 2 reads + 1 write → Option B ~25% less traffic, but reads may hit L2 anyway (A100 L2 40MB; with many rows streaming, each row's data might get evicted between passes... one CTA processes one row; pass 1 reads the row, pass 2 re-reads. Between passes, other CTAs are also streaming. If total working set across concurrent CTAs exceeds L2, pass 2 reads come from HBM. Rows per CTA: concurrent CTAs ~ up to 108 SMs * up to 16-32 CTAs... With 128 threads/CTA, each SM can host many CTAs (max 32 CTAs/SM on A100? Actually limit is 32 blocks per SM on A100... 64? For sm_80, max blocks per SM = 32). 108 SMs * 32 = 3456 concurrent CTAs. If N=100k, each row 400KB, concurrent footprint = 1.4GB >> L2. So pass 2/3 reads mostly from HBM. Option B (2 reads+1 write) vs Option A (3 reads + 1 write): traffic 12N vs 16N bytes → 25% faster potentially. Hmm, that's meaningful.

But wait — Option B pass 1 does 2 EXP per element. With expf being ~10 instructions and memory read being 1 instruction per 4 elements (128B per warp per 4-float vector)... Let me estimate arithmetic vs memory: per float element: Option B pass 1: load (amortized 1/4 instruction for 128-bit loads... actually with float4 loads, per element: ld.global.v4 per 4 elements), 2 EXP (each maybe 5-10 SASS instrs), plus fmaxf, fma. So ~15-25 instructions per element in pass 1, plus pass 2: load + EXP + mul + store. A100: 108 SMs * 64 FP32 lanes... memory bandwidth 1.55 TB/s (or ~2TB/s after upgrades). Instruction throughput: 108 SM * 64 cores * 1.41 GHz ≈ 9.7 TFLOP/s non-FFMA → ~9.7 T instr/s (FP32 pipe). Bytes: for Option B, 12 bytes/element → 1.55e12/12 ≈ 129 G elem/s. Instructions per element ~25-30 → 129G*27 ≈ 3.5 T instr/s < 9.7 T. OK so still memory bound even with 2 EXPs per element (assuming EXP ≈ 5-10 instrs). Good — Option B wins on traffic.

Hmm, but actually, is it worth the complexity? Let me also think about a variant: two-pass where pass 1 computes only max (1 read), pass 2 computes sum AND we would need pass 3 for write. That's Option A. Alternatively pass 1 = max+sum online (1 read), pass 2 = write (1 read + 1 write). Option B it is.

Wait, actually there's an even better trick when N is small enough to keep x in shared memory or registers... but "no dynamic shared memory" and static shared is capped (~48KB usable without opt-in; can go up to 163KB static? Actually static shared memory limit is 48KB per block by default; sm_80 allows up to 163KB with opt-in but that requires dynamic shared memory attribute). With 48KB static shared = 12288 floats. If N ≤ 12288, we could load x once into shared, then all passes read from shared: traffic = 1 read + 1 write. For N > 12288, fall back. The branch `if (N <= 12288)` depends only on N — allowed! Control flow depending on N is fine. That gives best of both: for small/medium N (common case), 1 read + 1 write and minimal EXP count (2 per element). For huge N, online 2-pass.

Hmm, but is that over-engineering? The requirement says "Aim for high performance (fast reductions, coalesced access)". A clean, fast, correct kernel is the goal. Let me think about what the grader likely wants: correct handling of arbitrary N, stability, block reductions with shuffles, coalesced loads, the EXP macro, data-independent control flow. Probably a 3-pass or online 2-pass with warp shuffles. 

Let me think about the shared memory idea more: static shared float tile[12288] = 48KB → limits occupancy to 1 block per SM (A100 has 164KB shared per SM; 48KB allows 3 blocks/SM). Hmm, 48KB static shared per block allows up to 3 CTAs/SM (164/48 = 3.4). That's OK-ish. But declaring 48KB static shared unconditionally reduces occupancy even for tiny N, hurting the small-N many-rows case which is occupancy/latency bound. Bad tradeoff. Could declare a smaller tile, e.g., 8KB (2048 floats), covering N ≤ 2048 fully in shared, else fallback. For N ≤ 2048, each thread handles ≤ 16 elements. Occupancy with 8KB shared: 164/8 = 20 blocks, not limiting (limit is 32 blocks/SM... also registers). Hmm.

But honestly, I think complexity is getting high. Let me evaluate the simple-but-solid design:

Design: online (single fused pass for max+sum) + write pass, using warp-level reductions. Data-independent control flow, coalesced vectorized loads where possible.

Vectorization: N arbitrary (not necessarily multiple of 4). We can do: main loop over float4 with bounds `i4*4+3 < N`... the tail handled scalar. Condition depends on index and N only — allowed. But misaligned rows: x row pointer = x + row*N — alignment of row start depends on N; if N is not a multiple of 4, row base isn't 16B aligned for later rows. float4 loads require 16B alignment → would misbehave/crash. To be safe: check alignment? That depends on pointer values (row index * N), which are not "data values" but the verifier might treat pointer arithmetic as fine since it depends on blockIdx and N... "Control flow may depend only on threadIdx/blockIdx/blockDim/N". Alignment of x + row*N depends on blockIdx, N, and the base pointer x. Base pointer alignment is unknown at compile time. Doing `if (((uintptr_t)p & 15) == 0)` — depends on the pointer, not on matrix data values. Risky for the verifier. Alternative: use __ldg / plain float loads with stride 128 — fully coalesced (each warp reads 128 consecutive floats = 512B per warp per iteration = 4 sectors... fine, coalesced). Scalar float loads at stride blockDim give perfect coalescing; the only cost is 4x more load instructions, which for a memory-bound kernel is typically fine (LSU throughput: 4 loads per cycle per SM for 32-thread warp... a warp ld.global.f32 issues 1 instruction and 1 memory transaction wavefront (128B per warp quarter? Actually 32 lanes * 4B = 128B = 1 wavefront... A100 LSU: 4 warp instructions per cycle? whatever). Scalar coalesced loads achieve full bandwidth in practice when enough warps in flight. expf-heavy loops might become instruction bound though.

Hmm, let me count instructions per element for Option B with scalar loads:

Pass 1 per element:
- ld.global (1/1 per element) — say amortized instruction issue 1 per element
- fmaxf: 1 (actually computed as max of running m: `nm = fmaxf(m, v)`: 1 FMNMX)
- EXP(m - nm): sub (FADD) + EXP → say EXP = 6 instrs (expf expansion) + 1 sub
- EXP(v - nm): 1 sub + 6
- s * that + ...: FMA 1, mul 1 → 2
Total ≈ 1 + 1 + 7 + 7 + 2 ≈ 18 instrs/element.

Pass 2 per element:
- ld: 1
- EXP(v - M): sub + 6 = 7
- mul by inv: 1
- st: 1
Total ≈ 10.

Grand total ≈ 28 instrs/element, 12 bytes/element.

A100: 108 SMs. Issue rate: 4 warp schedulers/SM, 1 instr/cycle each = 432 instrs/cycle total at ~1.41GHz → 609 G instr/s (warp-instructions). Per element per warp-lane: 28 warp instructions per 32 elements... wait let me redo: per element, each thread executes 28 instructions. Per warp, that's 28 warp-instructions per 32 elements (per lane-element). So elements/s = 609e9 * 32 / 28 = 696 G elem/s. Memory: 12 B/elem * 696e9 = 8.3 TB/s >> 1.55 TB/s. So instruction-bound, not memory bound! Hmm. That changes things — with expf being ~6-8 instructions, 2-3 EXPs per element makes us compute-bound.

Let me recount expf's SASS: expf(x) = exp2f(x * 1.442695) roughly, with exp2f using ex2.approx. Actually CUDA's expf: for |x| moderate, it's about: FMUL (x*log2e), then EX2? No — accurate expf handles range: it's implemented as `__internal_accurate_expf`? expf has max error ~2 ulp; implementation: multiply by log2(e), split into integer and fraction... I recall expf compiles to roughly 6-10 SASS instructions including EX2. __expf is 2 (FMUL by log2e, EX2). 

The EXP macro is from "common.cuh" — unknown implementation. Could be `#define EXP(x) __expf(x)` or `expf(x)`. Given "our formal verifier needs this" — it's likely just for verification instrumentation; performance-wise it's probably expf or __expf. I'll assume moderate cost.

So per-element cost: if EXP = __expf (2 instrs), Option B: pass1: 1(ld)+1(max)+1(sub)+2+1(sub)+2+2 ≈ 10, pass2: 1+1+2+1+1 = 6 → 16 instrs/elem, 12 B/elem → 696*... elements/s = 609e9*32/16 = 1218 G elem/s → 14.6 TB/s > 1.55 TB/s: memory bound. Good.

If EXP = expf (say 8 instrs), Option B: pass1: 1+1+1+8+1+8+2 = 22, pass2: 1+1+8+1+1 = 12 → 34 instrs/elem → 573 G elem/s * 12B = 6.9 TB/s > 1.55: still memory bound?? Wait: 609e9 warp-instr/s * 32 = 19.5e12 thread-instr/s; /34 = 573 G elem/s; bytes: 573e9*12 = 6.9 TB/s > 1.55 TB/s → memory bound. Hmm interesting, so even with expensive expf, Option B is memory bound at ~34 instrs/elem? Let me double check A100 issue: 108 SMs × 4 schedulers × 1.41 GHz = 609 G warp-instr/s. Yes. And FP32 pipe: 64 lanes/SM → 64/32 = 2 warp-instr/cycle/SM for FP32 → but issue is 4/cycle/SM across pipes; FP32 ops max 2 warp-instr/cycle/SM. Many of our instructions are FP32 (FADD, FMUL, FMNMX, FMA) plus EX2 (SFU: 16/SM/cycle → 0.5 warp-instr/cycle/SM for EX2 — SFU is the bottleneck for EXP!). 

SFU throughput on A100: 16 SFU lanes per SM → EX2 processes 32 threads in 2 cycles → 0.5 warp EX2 instr/cycle/SM. Total EX2 warp-instr/s = 108 * 0.5 * 1.41e9 = 76 G/s → per-lane: 2.43 G EX2-lane/s... wait: 76e9 warp-instr/s * 32 lanes = 2.43e12 EX2 ops/s. Option B needs 3 EX2 per element (2 in pass 1, 1 in pass 2) → 812 G elem/s max from SFU. Memory allows 1.55e12/12 = 129 G elem/s. So SFU not the bottleneck; memory is (129 G elem/s < 812 G). Hmm wait, that says memory bound at 129 G elem/s needing 129e9*3 EX2/s = 388 G EX2/s < 2.43 T/s. Fine.

And instruction issue: 34 instrs/elem at 129 G elem/s = 4.4 T thread-instr/s = 137 G warp-instr/s < 609 G/s. Fine. So memory bound either way. 

So Option B (online, 2 reads + 1 write = 12 B/elem) vs Option A (3 reads + 1 write = 16 B/elem): B is up to 33% faster in the HBM-bound regime. 

But actually wait — I should double check the read of x in pass 2 of Option B hitting L2/L1. The write pass re-reads the row right after pass 1. If the row is large (say 400KB), L1 is 192KB per SM (shared+L1), shared across up to 32 CTAs... won't be in L1. L2 40MB total across chip; the CTA's row data was just read; with 3456 concurrent CTAs each streaming 400KB, footprint 1.4GB >> L2 → pass-2 reads likely miss. So traffic really is 12 vs 16 B/elem. Option B better for large N.

For small N (row fits in L1/L2 easily, many rows): both fine; overheads dominated by reductions and launch.

Hmm, but wait — there's an even better option for reducing traffic: keep x values in registers! If each thread loads its elements into a fixed-size register array... N unknown at compile time. We could do: if N ≤ K*128, each thread has ≤ K elements; use a templated/loop-based approach with a compile-time max K and predication: `float v[K]; #pragma unroll for k: idx = tid + k*128; v[k] = (idx < N) ? x[idx] : -inf` — control flow via ternary select on index<N (index/N dependent, allowed; it's a select anyway). K would need to be, say, 32 → covers N ≤ 4096 with everything in registers, 1 read + 1 write + 2 EXP/elem... but register array indexing with unrolled loops and predication: `v[k] = (idx < N) ? __ldg(x+idx) : -FLT_MAX` — the load itself must be predicated: loads can't be "selected" without executing both sides... Actually `(idx<N) ? x[idx] : -FLT_MAX` compiles to a predicated load (LDG with predicate guard) — that's fine, predicated instructions are not branches; control flow (branch) stays uniform. The load is executed predicated-off when idx≥N — no memory access occurs. That's safe and standard. Then max over v[0..K-1] via fmaxf chain, block reduce, then EXP each, sum, block reduce, then write predicated. This keeps x in registers: traffic 1 read + 1 write = 8 B/elem, 2 EXP/elem. And for N ≤ K*128 it's super fast. Then fallback for larger N to the strided online approach.

Is the added complexity worth it? The problem says N is often not a multiple of 128 — suggesting they test various N. Performance target unclear. I think a well-implemented two-phase (registers for small N with K like 8 or 16, strided for large N) is genuinely how fast softmax kernels are written (e.g., with vectorized loads). But complexity risks bugs. Let me think about what maximizes correctness + decent performance + meets all constraints cleanly.

Honestly, I'll go with a clean, robust implementation:

1. Each thread processes elements with stride 128 (coalesced).
2. Online max+sum accumulation in registers in one pass over x (branchless rescale).
3. Block-level reduction: warp shuffle reduce within 4 warps (128 threads = 4 warps), then shared memory (static) exchange, then broadcast.
4. Combine per-thread (m_t, s_t) into global (M, S): M = max m_t; S = Σ s_t * EXP(m_t - M).
5. Second pass: y[i] = EXP(x[i] - M) * (1/S).

Wait, but there's a subtlety in step 4: I need each thread's own m_t after computing M to scale s_t. Order: reduce max of m_t → M (all threads get M via shared broadcast). Each thread computes s_t' = s_t * EXP(m_t - M). Then reduce sum of s_t' → S, broadcast. Two __syncthreads-separated reductions. Fine.

Alternative to avoid the scaling subtlety: pass 1 = max only; pass 2 = sum; pass 3 = write. Simpler, more memory traffic. Hmm.

Actually, hmm, wait. Let me reconsider. Is the online approach with per-thread rescaling numerically OK? Yes — it's exactly the flash-attention style parallel online softmax; standard.

But one concern: in pass 1, per element we do EXP(m - nm) where often m == nm (running max unchanged) → EXP(0) = 1, wasted work but correct. Fine.

Edge: N ≥ 1 but some threads see zero elements (N < 128): their m_t = -INF (init), s_t = 0. In combine: EXP(-INF - M). If M is finite: EXP(-inf) = 0, s_t * 0 = 0*0 = 0 fine. If M = -INF (only possible if all x are -INF): NaN — degenerate case, accept. Actually wait, can I avoid even that? Init m_t = -FLT_MAX instead of -INF. Then threads with no elements: m_t = -FLT_MAX, EXP(-FLT_MAX - M) underflows to 0 (if M finite), s_t=0 → 0. All-(-INF) row: M = -FLT_MAX (since fmaxf(-FLT_MAX, -INF) = -FLT_MAX), s_t = Σ EXP(-INF + FLT_MAX) = EXP(-inf) = 0 → S=0 → inv = inf → y = EXP(-INF + FLT_MAX) * inf = 0 * inf = NaN. Still NaN. Truly degenerate; every softmax implementation gives NaN or garbage for all -INF rows (even PyTorch gives NaN). Accept.

Hmm, actually also consider: what if N ≥ 1 but row has huge positive values, e.g., x ~ 100 (they said |x| ~ 100, fine) or even 1e30: EXP(1e30 - 1e30) = EXP(0) = 1 fine. Negative large: -1e30: EXP(-1e30 - (-1e30))=1. fine. Mixed ±1e30: max = 1e30, EXP(-1e30 - 1e30) = EXP(-2e30) = 0. Sum ≥ 1 (the max element gives EXP(0)=1). Stable. 

Denominator: sum could be large if many elements near max — up to N. No overflow (N ≤ ~2^31, sum ≤ N ≤ 4e9 < FLT_MAX). Actually if N ~ 1e9 and all equal, sum = 1e9, fine. inv = 1/sum, y = exp * inv ∈ [0,1]. Good.

Also: what about sum in float — for large N, sequential per-thread accumulation in float loses precision; typical and acceptable. Could use Kahan or per-thread accumulation is at most N/128 elements — fine.

Now, the block reduction details. blockDim = 128 → 4 warps. 

Warp reduce: for offset in {16,8,4,2,1}: val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset)). Then lane 0 writes to shared. __syncthreads. Warp 0 reads 4 values, reduces, writes result to shared[0] (or broadcast). __syncthreads. All read shared[0]. Or use __shfl_sync broadcast after. Two reductions (max for M, then sum for S) → can reuse the same shared arrays with syncs between. Total ~4 __syncthreads. Fine.

Shared memory: static __shared__ float sdata[4]; — tiny. "No dynamic shared memory" — static is allowed.

Full mask 0xffffffff for shuffles: all 128 threads participate always — no early exits, since control flow can't depend on data (and for reductions all threads must reach __syncthreads). All threads execute the loops with per-thread trip counts differing only by threadIdx/N — that's allowed (loop condition i < N depends on threadIdx and N). __syncthreads inside/after loops: all threads exit loops and reach the sync regardless — need to be careful that __syncthreads is NOT inside a loop whose trip count varies by thread! In my design, the reduction syncs happen after the loop completes for all threads — loop exit condition `i < N` where i = tid + k*128: different threads exit after different iterations, but they all eventually exit and reach the first __syncthreads. That's fine and standard (divergent loop trip counts are OK as long as syncs are outside the varying-trip-count region — actually CUDA requires all threads reach the same __syncthreads, which they do, just at different times; the hardware handles reconvergence at the barrier).

Hmm wait, one more consideration: for coalescing with stride-128 scalar loads — perfect coalescing. Good enough. Should I vectorize with float4 when N % 4 == 0 and pointer aligned? Pointer alignment unknown → skip; scalar is fine for correctness. But performance... Let me think about whether scalar loads limit bandwidth. Per warp per iteration: one 128B transaction (32 lanes × 4B contiguous). That's fully efficient — the memory system sees perfectly coalesced 128B requests. The instruction count is 4× higher than float4 loads, but we computed we're not instruction-bound (34 instrs/elem worst case vs budget 609e9*32/129e9 = 151 instrs/elem... wait recompute: at memory-bound 129 G elem/s, instruction budget = 19.5e12 thread-instr/s ÷ 129e9 elem/s = 151 instrs/elem. We use ~34. Plenty of headroom.) So scalar loads are fine. 

Now, ILP: each thread accumulates a single running (m, s) with a loop-carried dependency: s = s*EXP(m-nm)+EXP(v-nm) — the FMA depends on previous s and m. This serializes per-thread progress: latency chain per element ≈ (max + sub + exp + fma + mul + exp...) ~ 30-50 cycles per element?! With 32 CTAs × 4 warps... occupancy: 128 threads/CTA; registers modest (~40) → CTAs/SM limited by max 32 blocks/SM? A100 supports 32 blocks/SM max... wait actually for sm_80 it's 32 blocks per SM. 32 blocks × 4 warps = 128 warps... no wait, max warps/SM = 64. 128 threads = 4 warps/CTA; 64/4 = 16 CTAs/SM (if registers/shared allow). 16 CTAs × 128 = 2048 threads = 64 warps/SM. Each warp: the dependency chain per element is ~2 EXPs + fma ≈ maybe 40-60 cycles. Per-element throughput per SM = 64 warps × 32 lanes / 50 cycles ≈ 41 elem/cycle/SM... needed: 129e9 elem/s ÷ (108 SM × 1.41e9) = 0.85 elem/cycle/SM. Way below 41. Fine — latency hidden by occupancy.

Alternative to reduce EXP count in pass 1: two-pass over registers? No.

Actually, you know, let me reconsider Option A vs B once more for typical test sizes. If the harness tests N in the hundreds-to-thousands with many rows (like a language model logits matrix: rows=vocab... e.g., rows=32000, N=4096? Actually GLM! "softmax_glm" — GLM the language model. Row-major x[rows, N] — maybe rows = batch*seq, N = vocab (e.g., 150k for GLM-130B?) Or transposed. Either way, both large-N and many-rows regimes plausible. With grid=rows fixed, if rows is small and N huge (e.g., rows=32, N=150000), only 32 CTAs on 108 SMs — underutilized, memory latency bound; 3-pass would be 1.5× more traffic but each pass has more parallelism... eh, both are latency-bound there; fewer passes better regardless.

If rows huge and N small (e.g., rows=1e6, N=100): launch-bound; passes over L2-resident data; reduction overhead dominates; both similar (B slightly better: 2 passes vs 3).

OK going with Option B (online). Hmm, wait, actually let me reconsider once more — is there any correctness risk with online that I should double check? The rescale math: thread-local: m running max, s = Σ exp(v - m). Update with new v: nm = max(m, v); s' = s * exp(m - nm) + exp(v - nm). If m == nm: s' = s + exp(v - m). Correct. If v > m: s' = s * exp(m - v) + 1. Correct. Initial m0 = -INF, s0 = 0: s' = 0 * exp(-INF - nm) + exp(v - nm) = 0*0 + exp(v-nm) = 1 if nm = v. Wait 0 * exp(-INF - v): exp(-inf)=0, 0*0=0 ✓. But if m0 = -INF and nm = -INF (v = -INF): exp(-INF - -INF) = exp(NaN) = NaN → s becomes NaN! With m0 = -FLT_MAX and v = -INF: nm = fmaxf(-FLT_MAX, -INF) = -FLT_MAX; s' = 0 * exp(-FLT_MAX + FLT_MAX) + exp(-INF + FLT_MAX) = 0*exp(0)=0 + exp(-inf)=0 → 0. Good, no NaN. So init m = -FLT_MAX handles -INF inputs gracefully (outputs all... sum=0 → NaN, degenerate anyway but at least defined). Hmm, but what if ALL elements are -FLT_MAX (max = -FLT_MAX): exp(0)=1 each, fine, normal. Good: init with -FLT_MAX.

Hmm wait, actually there's still a subtle NaN risk with -FLT_MAX init in the combine step when a thread had no elements: s_t = 0, m_t = -FLT_MAX; scale = EXP(m_t - M) = EXP(-FLT_MAX - M). If M = -FLT_MAX (possible when all real elements are -INF or ≤ -FLT_MAX... or when the row's max is -FLT_MAX and threads without elements also -FLT_MAX): EXP(0) = 1 → contribution 0*1 = 0 ✓. If M finite: EXP(-huge) = 0 → 0*0 = 0 ✓. Good, no NaN from empty threads.

What about M = -FLT_MAX exactly and other threads with elements = -INF: s_t = 0 → 0. S = Σ... the max element thread: element -INF? If max is -FLT_MAX, all elements ≤ -FLT_MAX... elements could be -INF → s_t = 0; elements = -FLT_MAX → s_t = count. S ≥ 1 if any element is finite ≥ -FLT_MAX. Only all-(-INF) rows give S=0 → NaN output. Degenerate, fine.

Alright, the algorithm is solid. Now let me also handle the write pass: y[i] = EXP(x[i] - M) * inv. inv = 1.0f / S computed once per thread (or __frcp). Use inv multiplication (faster, and division once per row is fine either way; 1/S then multiply — slight precision difference, standard).

Now the reduction implementation details:

```
__shared__ float sred[4];       // warp partials
__shared__ float bcast[2];      // final M and S? or reuse
```

Warp max reduce:
```
unsigned mask = 0xffffffffu;
#pragma unroll
for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(mask, v, off));
```
Using xor (butterfly) so all lanes have the result — then lane 0 of each warp writes to shared. Actually with xor, every lane ends with the warp result; only lane 0 writes. Then sync; thread 0..3 (or warp 0) reads sred[0..3], reduces with xor shuffles within warp 0 (need only 4 values: lanes 0-3), writes final to shared; sync; all threads read. 

Cleaner: 
```
// reduce within warp (all lanes get result)
v = warp_reduce_max(v);
if (lane == 0) sred[warp_id] = v;
__syncthreads();
// thread 0 combines 4 values
if (tid == 0) {
    float r = fmaxf(fmaxf(sred[0], sred[1]), fmaxf(sred[2], sred[3]));
    bcast[0] = r;
}
__syncthreads();
float M = bcast[0];
```
Or have warp 0 do it. Simple enough. For the sum reduction same pattern with +.

Alternatively a single generic reduce function; but I need both max and sum — write two small helpers or one templated device function. Since kernel must be self-contained with common.cuh include, I'll write inline device functions.

Note: __syncthreads count: pass1 → reduce max (sync, sync) → scale → reduce sum (sync, sync) → write pass. 4 syncs. OK.

Can I combine: reduce both m (max) and s' (after scaling)... no — scaling needs M first. Alternative: reduce pairs via two shared rounds: Actually there's a trick: first reduce max; then each thread scales; then reduce sum. Two reductions necessary. Fine.

Hmm, alternatively — avoid online entirely and do 3-pass but with fewer syncs? 3-pass: max reduce (2 syncs), sum reduce (2 syncs), write. Same sync count as online but an extra global read pass. Online is better.

Wait, actually, here's another thought: could do pass 1 = max only (no sum), pass 2 = sum + recompute? No — write pass needs both M and S; sum pass needs M. So 3 passes. Online: 2 passes. Confirmed.

Let me now also think: should I unroll with multiple accumulators for ILP in pass 1? The online recurrence is serial per thread. Could process 2 elements per iteration with independent chains then combine: (m1,s1) and (m2,s2) chains for even/odd elements, combine at end: m = max(m1,m2); s = s1*EXP(m1-m) + s2*EXP(m2-m). This doubles ILP at cost of 2 extra EXPs per thread total (negligible) and more registers. Given we're memory/latency bound with plenty of occupancy headroom, probably unnecessary. But for the large-N-few-rows case (underutilized GPU, few CTAs), ILP matters: with rows=32, only 32 CTAs = 128 warps total across the GPU (1-2 warps per SM!). Then per-element serial latency dominates: chain ≈ fmaxf(4c) + sub(4c) + EXP(~15c?) + fma(4c) + ... ~30-40 cycles per element per thread. Per CTA: 4 warps × 32 lanes / 35 cycles ≈ 3.7 elem/cycle... but only ~1 CTA per SM → 3.7 elem/cycle/SM vs memory capability ~... DRAM latency-bound: each warp iteration requests 128B; with 4 warps in flight per SM and ~500ns latency... Little's law: to saturate 1.55TB/s need ~750KB in flight; 32 CTAs × 4 warps × 128B = 16KB in flight → ~2% of bandwidth → totally latency bound. So for few-rows case, we want max ILP: multiple elements in flight per thread. 2 or 4 independent online chains would help ~2-4×. Also prefetching next load before computing current (software pipelining) helps.

Let me design pass 1 with 4 independent (m, s) chains? That's 8 registers for accumulators + addressing. Hmm, but with stride-128 loads, thread t accesses t, t+128, t+256... — for 4-way ILP we can just have the loop process 4 elements per iteration: indices i, i+128, i+256, i+512, then stride 512. Coalescing still perfect (consecutive threads → consecutive addresses). Each of the 4 element streams has its own (m_k, s_k) chain. Trip count varies per thread (i + 3*128 < N etc. — bound conditions on index & N only). Tail: handle remaining 0-3 streams... messy but doable: loop `for (base = tid; base < N; base += 512)` and inside, 4 predicated loads at base, base+128, base+256, base+384? But then chains' element assignment: chain k gets base + k*128 for each base — each chain sees increasing indices stride 512. Fine.

Predication: `float v = (idx < N) ? __ldg(&x[idx]) : -FLT_MAX;` — hmm wait, but for the sum, adding EXP(-FLT_MAX - nm) would add exp(-huge)=0, not affecting the sum ✓ as long as nm is finite. But nm = max(m, -FLT_MAX)... if the real elements are all < -FLT_MAX? Impossible: -FLT_MAX is the lowest finite float. If v = -FLT_MAX (padding) and chain's current m = -FLT_MAX (nothing real yet): nm = -FLT_MAX; s += EXP(0) = 1! Padding would contribute 1 to the sum!! Bug. Need padding that never contributes: use -INFINITY for padding → EXP(-INF - nm) = 0 for finite nm ✓; and fmaxf(m, -INF) = m keeps chain clean; and if chain has only padding: m stays -FLT_MAX(init)... wait if I init m = -FLT_MAX and pad with -INF: nm = fmaxf(-FLT_MAX, -INF) = -FLT_MAX; EXP(-INF - (-FLT_MAX)) = EXP(-INF) = 0 ✓. s unchanged ✓. And EXP(m - nm) with m = nm = -FLT_MAX: EXP(0) = 1, s*1 ✓. OK: pad value -INFINITY works with init -FLT_MAX. But hmm — the select `(idx < N) ? load : -INFINITY` — is a select on index vs N (allowed, not data). The load must be predicated to avoid OOB access — compiler generates predicated LDG for the ternary? For `cond ? __ldg(p) : c`, the compiler can emit a predicated load (it does: `@p LDG` then SEL). I believe NVCC generates predicated loads for such patterns. To be extra safe, could write `float v = -INFINITY; if (idx < N) v = __ldg(&x[idx]);` — an if on (idx < N) — condition depends on threadIdx and N only, NOT on data. That's explicitly allowed: "Control flow may depend only on threadIdx/blockIdx/blockDim/N". So `if (idx < N)` is fine and unambiguous for the verifier! Data values never control flow. 

That simplifies: use explicit `if (idx < N)` guards for loads/stores. Predication/branches on index bounds are permitted.

So the plan crystallizes:

```cuda
#include "common.cuh"

#define TID (threadIdx.x)
#define NT 128
#define WARPS 4

__inline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
```

Block reduce: 
```cuda
__shared__ float smax[WARPS];
__shared__ float ssum[WARPS];
__shared__ float bcast; // result broadcast
```
Hmm, two shared arrays for warp partials (can reuse one for both reductions sequentially, but need separate broadcast slot... can reuse too with syncs). Let me write a clean pattern:

```cuda
__inline__ __device__ float block_max(float v) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smax[wid] = v;
    __syncthreads();
    // let warp 0 combine
    if (wid == 0) {
        float t = (lane < WARPS) ? smax[lane] : -FLT_MAX; // hmm, lane<WARPS depends on threadIdx — fine
        t = warp_reduce_max(t);
        if (lane == 0) smax[0] = t; // reuse slot 0... but careful: others read smax[0] after sync
    }
    __syncthreads();
    return smax[0];
}
```
Wait, reusing smax[0] as the broadcast slot: warp 0 lanes read smax[0..3] BEFORE writing smax[0] — within warp 0, after warp_reduce over the register copies, lane 0 writes smax[0]. Other warps read smax[0] after __syncthreads. Race? Warp 0 reads smax[lane] (lane<4) at start; writes smax[0] at end; no other warp writes smax between the two syncs. The write by lane 0 of warp 0 and reads by other lanes of warp 0 of smax[0]?? In warp_reduce, values are in registers, not re-reading shared. So only lane 0 writes smax[0] after all warp-0 lanes have loaded smax[lane] into registers. But wait — is there a guarantee that lane 0's write to smax[0] happens after lane 1..3's reads of smax[lane]? Within a warp, the initial loads `t = (lane<4)?smax[lane]:...` execute as one warp instruction (all lanes together), then shuffles, then lane 0's store. Warp executes in lockstep-ish (Volta+ independent scheduling, but data dependency via __shfl_sync synchronizes: the shuffle has sync semantics; the store depends on shuffle result). Reads of smax by lanes 1-3 happen before the shuffles; the store happens after. So no race within warp 0. And other warps only read smax[0] after the second __syncthreads. Safe. But subtle — to be safer and clearer, use a separate broadcast variable:

```cuda
__shared__ float s_part[WARPS];
__shared__ float s_final;
```
Fine, trivial shared usage.

Alternatively simpler: since only 4 warps, after first sync, every thread reads all 4 partials and computes the final locally:
```cuda
if (lane == 0) s_part[wid] = v;
__syncthreads();
float r = fmaxf(fmaxf(s_part[0], s_part[1]), fmaxf(s_part[2], s_part[3]));
```
Every thread computes r from shared — no second sync needed, no broadcast slot, no warp-0 special path. Clean and fast. 

Similarly for sum: r = s_part[0]+s_part[1]+s_part[2]+s_part[3].

Now the main kernel:

```cuda
extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid = threadIdx.x;          // 0..127
    const float* __restrict__ xr = x + (size_t)blockIdx.x * N;
    float* __restrict__ yr = y + (size_t)blockIdx.x * N;
    ...
}
```

size_t cast for row offset: rows*N could exceed 2^31 (e.g., rows=1e6, N=4096 → 4e9 > 2^31). blockIdx.x * N: blockIdx.x is unsigned int (max 2^31-1 grid x dim... gridDim.x limit is 2^31-1), N int. Product as unsigned 32-bit could overflow. Use (size_t)blockIdx.x * (size_t)N. Yes, important for correctness with huge matrices. 

Pass 1 (online max+sum), with 2 or 4 independent chains for ILP. Let me think about whether to bother. Registers: 4 chains × 2 = 8 regs + temps. Let me do 4-way: indices tid, tid+128, tid+256, tid+384, then stride 512.

Wait, actually — hmm, with 4 chains each handling stride-512 elements starting at tid + k*128: chain k processes indices {tid + k*128 + j*512}. Coalescing: at iteration j, the 128 threads load {tid + k*128 + j*512} for tid=0..127 → contiguous 128-float block at k*128 + j*512. All four loads in an iteration are coalesced. ✓.

Loop:
```cuda
float m0=-FLT_MAX, m1=-FLT_MAX, m2=-FLT_MAX, m3=-FLT_MAX;
float s0=0, s1=0, s2=0, s3=0;
int i0 = tid, i1 = tid+128, i2 = tid+256, i3 = tid+384;
for (; i0 < N; i0 += 512, i1 += 512, i2 += 512, i3 += 512) {
    // load 4 values, predicated on i_k < N
    float v0 = (i0 < N) ? xr[i0] : -INFINITY; // hmm, need predicated load...
```
Careful: the loop condition only guards i0. Within the iteration, i1, i2, i3 might be ≥ N. Guard each. Using ternary might not predicated-load safely? NVCC will generate predicated/conditional load for `cond ? xr[i] : c` — it must, because unsafe memory access otherwise... Actually no: the compiler is allowed to speculate loads only if it can prove safety? For a ternary expression, C++ semantics say the load only happens if selected?? No! C++ ternary: `cond ? *p : c` — if cond is false, *p is not evaluated... it's like an if. The compiler must ensure no fault when cond false — it emits a predicated load or a branch. Either way, correct. OK so ternary is safe semantically. But does it emit a branch (divergent per-thread)? Within a warp, the condition i1 < N is uniform-ish near the tail only; mostly all-true. Compiler typically emits predicated LDG for short bodies. Either way correct; and condition depends on threadIdx/N only — verifier-safe.

Hmm wait, actually, let me reconsider whether the verifier is OK with ternary on data? "Control flow may depend only on ... never on data values (use fmaxf(a,b), not if (a>b))". Our ternaries/ifs are on index bounds — fine.

Then update chains:
```cuda
    nm = fmaxf(m0, v0); s0 = s0 * EXP(m0 - nm) + EXP(v0 - nm); m0 = nm;
```
Wait — EXP(m0 - nm) when m0 = nm = -FLT_MAX → EXP(0) = 1 → s0 = 0*1 + EXP(-INF + FLT_MAX) = 0 + 0 = 0 ✓ (v0 = -INF padding). Good. When m0=-FLT_MAX (chain empty) and v0 real: nm = v0 (if v0 > -FLT_MAX, always true for finite); EXP(-FLT_MAX - v0) = 0 → s0 = 0*0 + EXP(0) = 1 ✓.

Per-element EXP count: 2. For 4 chains over all elements: 2 EXP/element in pass 1. Plus 1 in pass 2. Total 3 EXP/element. Hmm! That's the cost of online. 3-pass approach: 2 EXP/element but 4 accesses (3R+1W = 16B) vs online 3 EXP/element, 12B. If EXP is cheap (SFU EX2-based), online wins for memory-bound. As computed above: memory bound at 129 Gelem/s; EXP capacity 2.43 Telem-EX2/s... 3 EXP/elem → 388 G EXP/s needed < 2430 G ✓. Instruction issue: ~40 instrs/elem → 5.2 T thread-instr/s at 129 Gelem/s... wait 129e9 × 40 = 5.16e12 < 19.5e12 ✓. So online is memory-bound ✓, 3-pass would be memory-bound at 16B → 97 Gelem/s. Online 33% faster in the streaming regime. 

But hold on — is the online 2-EXP-per-element in pass 1 really necessary? Alternative: within pass 1, per-thread compute just max (1 EXP/0 EXP), and... no, online requires the rescale EXP. Alternatively: pass 1 computes max with 4 chains (no EXP); pass 2 computes sum with 4 chains (1 EXP/elem); pass 3 writes (1 EXP/elem): total 2 EXP/elem, 16B/elem. Online: 3 EXP/elem, 12B/elem. Memory bound either way → online wins on bandwidth. Unless EXP is very expensive (expf accurate ~ maybe 12-16 instrs?): then instr/elem online ≈ 2×(16)+16+overhead ≈ 55; at 129 Gelem/s → 7.1 T thread-instr/s < 19.5 T ✓ still fine. SFU: expf uses one EX2 each → 3 EX2/elem → 388 G/s ✓. So online stays memory-bound. Go online.

Hmm, although... wait. Let me reconsider: maybe the harness runs with moderate N (like ≤ 4096) where rows fit in L2, many rows. Then traffic is not purely HBM; reads hit L2 (L2 bandwidth ~5-7 TB/s on A100). Then instruction/EXP cost matters more, and 3-pass's extra read is cheap (L2 hit). Ugh, hard to know. The online approach is strictly ≥ in performance in nearly all regimes (its only cost is 1 extra EXP per element, and EXP is cheap on SFU). I'm confident in online.

Actually hmm, wait, one more idea to cut pass-1 EXPs: instead of rescaling per element, note that we could do per-thread: first find thread-local max over its elements (no EXP), then sum EXP(v - m_t) (1 EXP/elem), then combine across threads with the scaling trick (needs each thread's m_t — we have it). That's exactly 1 EXP/elem in pass 1 + 1 in pass 2 = 2 EXP/elem total, AND only 2 passes (12B/elem)! This requires each thread to make TWO passes over ITS OWN elements — but its elements aren't stored... it would re-read x from memory (that's the 3-pass = 16B). UNLESS the thread's element count is small enough to keep in registers! 

Oh wait — here's the thing: each thread handles ceil(N/128) elements. For N ≤ 4096, that's ≤ 32 elements/thread — 32 registers, too many? 32 floats = 32 regs just for data + ~30 others → ~64 regs, occupancy 50%... borderline. For N ≤ 2048: 16 regs → fine. Register caching with compile-time max K and predication: works only for N ≤ K*128; else fallback to online strided. Two code paths — both must be correct; more testing surface. Complexity ↑. 

Decision: keep it simple and robust — pure online approach with 4 ILP chains, 2 passes. It's clean, correct for any N, good performance in all regimes. Actually hmm, let me reconsider the number of chains: 4 chains = 4× unrolled strided loads. For coalescing: iteration j loads 4 contiguous 512B blocks (each 128 threads × 4B). Fine.

But wait — there's a subtlety with 4 chains and N < 512: e.g., N = 300: i0 ∈ [0..127] loads at tid<... i0 = tid < 300 → all 128 threads load (i0 ≤ 127 < 300 ✓). i1 = tid+128 ∈ [128..255] < 300 ✓ all load. i2 = tid+256 ∈ [256..383]: threads with tid < 44 load; others pad. i3 = tid+384 ≥ 384 > 300 → nobody loads. Loop: i0 starts tid < 300... i0 = tid ≤ 127 < 300 → enter loop once; after +=512, i0 = tid+512 > 300 → exit. ✓ Each element visited exactly once: element index e is handled by thread e%128, chain... e = tid + k*128 + j*512 with tid = e%128, then (e - e%128)/128 = k + 4j → k = ((e div 128) mod 4), j = (e div 512). ✓ Exactly once. 

Chain combine at end of pass 1:
```cuda
float m = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
float sc0 = s0 * EXP(m0 - m); // EXP(-FLT_MAX - m) = 0 if m finite; if m = -FLT_MAX... 
```
Edge: chains with no elements: m_k = -FLT_MAX, s_k = 0. If overall m (thread's max) is finite: EXP(-FLT_MAX - m) = 0 → 0*0 = 0 ✓. If thread's m = -FLT_MAX (thread had no real elements OR all its elements ≤ -FLT_MAX, i.e., = -FLT_MAX or -INF): chains with s_k=0: EXP(0)=1 → 0*1=0 ✓; chains with real elements = -FLT_MAX: s_k = count, EXP(0)=1 → s_k ✓. Chains with -INF elements: m_k = -FLT_MAX, s_k = 0 ✓. What if thread's elements are all -INF: m = -FLT_MAX; combine: all s_k = 0 → 0. ✓. Good — combine formula: s = Σ_k s_k * EXP(m_k - m) with m = max_k m_k. When m_k = m: EXP(0)=1 ✓. When m_k < m: EXP(neg) ≤ 1 ✓. When m_k = -FLT_MAX and m = -FLT_MAX: 1, times s_k=0 ✓. 

Then block-reduce m (max) → M; scale s by EXP(m - M) — wait, thread's s is now relative to thread's m. s_scaled = s * EXP(m - M). Then block-reduce sum → S.

Edge in s * EXP(m - M): m = -FLT_MAX (empty thread), M finite: EXP(-huge) = 0, s = 0 → 0 ✓. m = -FLT_MAX, M = -FLT_MAX (row max is -FLT_MAX or row all -INF): EXP(0)=1, s=0 → 0 ✓. Row all -INF: S=0 → 1/0 = inf → y = EXP(-INF - -FLT_MAX)*inf = 0*inf = NaN. Degenerate; fine.

Another subtle: what if M = +INFINITY (input has +INF)? EXP(v - M): for v finite: EXP(-inf) = 0; for v = +INF: EXP(nan) = NaN → NaN in sum → y NaN. Degenerate input; whatever. Actually hmm, x ~ 100 is their stability test; ±INF inputs presumably not tested.

Now precision: per-thread partial sums with rescaling — error accumulates ~ same as any online softmax; standard and fine for float32.

Pass 2: 
```cuda
float inv = 1.0f / S; // or __frcp_rn? plain division fine (once per row... per thread actually)
for (int i = tid; i < N; i += 128)
    yr[i] = EXP(xr[i] - M) * inv;
```
Wait — division: `1.0f / S` per thread (all compute same). Fine. Should I use __fdividef? Plain is IEEE div — costs a few extra instrs once per row; negligible.

Coalescing ✓. Could also 4-way unroll pass 2 with chains for ILP:
```cuda
i0=tid; ... for(; i0<N; i0+=512){ if(i0<N) yr[i0]=EXP(xr[i0]-M)*inv; ... }
```
Since EXP has ~10+ cycle latency and the store depends on it, 4 independent chains help in latency-bound regimes. Let me do 4-way in pass 2 as well, or at least 2-way. Actually pass 2 per element: load → sub → EXP → mul → store; dependency chain ~25-40 cycles per element per thread; with 4 chains → ~7-10 cycles/element. With 16 CTAs/SM × 4 warps × 32 lanes / 8 cyc ≈ 256 elem/cycle/SM — way beyond need when memory bound (need ~1). For latency-bound (few rows), 4-way helps 4×. I'll do 4-way in both passes. Hmm, but code duplication for guards... manageable with a helper macro or just careful code.

Alternatively write pass 2 as simple stride loop — but let me keep ILP. Actually, let me think about clean code structure:

Pass 1:
```cuda
float m0 = -FLT_MAX, m1 = -FLT_MAX, m2 = -FLT_MAX, m3 = -FLT_MAX;
float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

int i = tid;
for (int base = tid; base < N; base += 4 * BLOCK) {
    float v0 = (base + 0*BLOCK < N) ? xr[base + 0*BLOCK] : -INFINITY;
    float v1 = (base + 1*BLOCK < N) ? xr[base + 1*BLOCK] : -INFINITY;
    float v2 = (base + 2*BLOCK < N) ? xr[base + 2*BLOCK] : -INFINITY;
    float v3 = (base + 3*BLOCK < N) ? xr[base + 3*BLOCK] : -INFINITY;

    float nm;
    nm = fmaxf(m0, v0); s0 = s0 * EXP(m0 - nm) + EXP(v0 - nm); m0 = nm;
    nm = fmaxf(m1, v1); s1 = s1 * EXP(m1 - nm) + EXP(v1 - nm); m1 = nm;
    nm = fmaxf(m2, v2); s2 = s2 * EXP(m2 - nm) + EXP(v2 - nm); m2 = nm;
    nm = fmaxf(m3, v3); s3 = s3 * EXP(m3 - nm) + EXP(v3 - nm); m3 = nm;
}
```

Hmm wait, there's an issue: `base + k*BLOCK < N` — base < N guaranteed by loop; these are int comparisons — fine, depends on tid/N.

But hmm — one concern about the ternary producing predicated loads: for `float v0 = (idx < N) ? xr[idx] : -INFINITY;` the compiler will emit: predicated LDG + SEL (or a branch). It's correct either way. OK.

But actually, wait: is `-INFINITY` OK to appear as EXP argument when nm is also -FLT_MAX... covered above: EXP(-INF - (-FLT_MAX)) = EXP(-INF) = 0 ✓. But hmm, what about v = -INF padding with chain m_k = -FLT_MAX and nm = fmaxf(-FLT_MAX, -INF) = -FLT_MAX ✓ (fmaxf returns the larger; -INF < -FLT_MAX). So nm = -FLT_MAX, EXP(v - nm) = EXP(-INF + FLT_MAX) → exponent -inf → 0 ✓. And EXP(m - nm) = EXP(0) = 1, s*1 + 0 ✓.

Also what if N < 128? Then some threads have i0 = tid ≥ N: loop body never executes (base = tid ≥ N). Their chains stay (-FLT_MAX, 0). ✓.

Pass 1 combine chains:
```cuda
float m = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
float s = s0 * EXP(m0 - m) + s1 * EXP(m1 - m) + s2 * EXP(m2 - m) + s3 * EXP(m3 - m);
```
Wait, careful: EXP(m0 - m) when m0 = -FLT_MAX, m = -FLT_MAX → EXP(0) = 1, s0 = 0 → 0 ✓. When m0 = -FLT_MAX, m finite → EXP(-huge) = 0, s0 = 0, 0*0 = 0 ✓. But: 0 * 0 = 0 ✓, and 0 * 1 = 0 ✓. No NaN since we never do 0 * inf or inf - inf... m0 - m: both -FLT_MAX → 0 ✓; -FLT_MAX - finite → large negative finite ✓ (not -inf... -FLT_MAX - M: if M = +1e38? -3.4e38 - 1e38 = -4.4e38 → overflows float to -INF → EXP(-INF) = 0 ✓ still fine). OK.

Hmm, but actually — could s0 * EXP(m0 - m) produce NaN if EXP(...) = 0 and s0 = inf? s0 inf impossible (sums bounded by count). ✓.

Block reduce:
```cuda
// max across block
m = warp_max(m);
__shared__ float wpart[4]; // warp partials
if ((tid & 31) == 0) wpart[tid >> 5] = m;
__syncthreads();
float M = fmaxf(fmaxf(wpart[0], wpart[1]), fmaxf(wpart[2], wpart[3]));
// every thread computes M locally from shared — no second sync needed yet
// scale own sum to global max reference
float sScaled = s * EXP(m_local - M);
```
Wait — careful! After computing M, each thread needs ITS OWN m (pre-reduction) for scaling. So keep local m in a separate variable. Let me name: lm (local max), ls (local sum, relative to lm). After reduction: M = block max. s' = ls * EXP(lm - M). Then reduce sum:

```cuda
float ls2 = warp_sum(s');
if ((tid & 31) == 0) wpart[tid >> 5] = ls2;
__syncthreads();
float S = wpart[0] + wpart[1] + wpart[2] + wpart[3];
```
Can reuse wpart between the two reductions? First reduction writes wpart, syncs, all read → compute M and s'. Then second reduction: warp-sum writes wpart again — but WAIT: have all threads finished READING wpart[0..3] (for M) before some warp overwrites wpart in the second reduction? There's no sync between "all threads compute M from wpart" and "lane0 writes wpart[wid] = ls2". Race! Thread A (warp 2) could still be computing M = fmax(wpart...) while thread B (warp 0 lane 0) already wrote wpart[0] = ls2. Need a __syncthreads between: after computing M (all threads read wpart), sync, then write wpart for sum reduction. So:

```cuda
// reduction 1 (max)
lm = warp_reduce_max(lm);
if (lane == 0) wpart[wid] = lm;
__syncthreads();
float M = fmaxf(fmaxf(wpart[0], wpart[1]), fmaxf(wpart[2], wpart[3]));
float sc = ls * EXP(lm - M);
__syncthreads();               // ensure everyone read wpart before overwrite
// reduction 2 (sum)
sc = warp_reduce_sum(sc);
if (lane == 0) wpart[wid] = sc;
__syncthreads();
float S = wpart[0] + wpart[1] + wpart[2] + wpart[3];
```
Total 4 __syncthreads (well, 3 here + 1 more? count: after write wpart (1), before overwrite (2), after write wpart (3)). Plus loop overhead. Fine.

Hmm, wait: is the second sync placed correctly? Sequence: sync1 → all threads read wpart → compute M, sc (no shared access) → sync2 → lane0s write wpart → sync3 → all read wpart → S. ✓ correct. Could use two separate shared arrays to drop sync2 (wpart_max[4] and wpart_sum[4]) — saves one sync. Shared memory is free (8 floats). Let me use two arrays: smax_part[4], ssum_part[4]. Then:

```cuda
lm = warp_reduce_max(lm);
if (lane == 0) smax_part[wid] = lm;
__syncthreads();
const float M = fmaxf(fmaxf(smax_part[0], smax_part[1]), fmaxf(smax_part[2], smax_part[3]));
float sc = ls * EXP(lm - M);
sc = warp_reduce_sum(sc);
if (lane == 0) ssum_part[wid] = sc;
__syncthreads();
const float S = ssum_part[0] + ssum_part[1] + ssum_part[2] + ssum_part[3];
```
3 syncs total, no race: smax_part written before sync1, read after sync1 by everyone — but is it overwritten later? No, only written once. ssum_part written after... wait: after sync1, threads read smax_part (fine), then compute sc, warp-reduce, then lane0 writes ssum_part[wid] — ssum_part is a DIFFERENT array, no conflict with smax_part reads. But do all threads finish reading smax_part before...? smax_part is never written again. ✓. ssum_part: written by lane0s (after their own reads of smax_part — different array), then sync2, then read by all. ✓. But hold on: is there a WAR hazard: thread X still reading smax_part[2] while thread Y writes ssum_part[0]? Different addresses — no hazard. ✓ 2 syncs. 

Then pass 2 with M and inv = 1/S.

Wait, one more consideration on the warp reductions: __shfl_xor_sync with mask 0xffffffff — all 32 lanes of each warp active? Yes: no thread exits early; every thread executes the full kernel body (loops have per-thread trip counts but all threads reach the reduction code). At the point of shuffle, all lanes are active... hmm, actually with Volta+ independent thread scheduling, threads within a warp might be at different program points when trip counts differ! If threads 0-31 have different loop trip counts (N not multiple of 512), they exit the loop at different iterations — but they converge at the loop exit... do they? With ITS, there's no guarantee of automatic reconvergence, BUT __shfl_xor_sync(0xffffffff, ...) itself synchronizes the named lanes: it waits for all lanes in the mask to arrive at the shuffle. So even if lanes arrive at different times, the shuffle with full mask forces convergence. ✓ Standard practice. And __syncthreads similarly. ✓.

Also, for correctness of the loop-exit: all threads eventually exit the loop (finite trip counts) ✓.

Now, pass 2. Write: 
```cuda
const float inv = 1.0f / S;
for (int base = tid; base < N; base += 512) {
    if (base + 0 < N) yr[base] = EXP(xr[base] - M) * inv; ...
}
```
Hmm, with 4-way:
```cuda
for (int b = tid; b < N; b += 512) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int idx = b + k*128;
        if (idx < N) yr[idx] = EXP(xr[idx] - M) * inv;
    }
}
```
Wait, but the loads `xr[idx]` here: should I use __ldg? x is const __restrict__ → compiler emits LDG anyway (or LDG.E.128... scalar: LDG.E with .nc? With __restrict__ const, nvcc uses non-coherent loads automatically? It does for __restrict__ const pointers — actually it requires __ldg or const __restrict__ which we have). Fine.

Alternatively, unroll pass 2 with explicit k=0..3 and separate ifs — same thing. 

Alternatively, simpler pass 2 (stride-128 loop, no unroll): for latency hiding relies on occupancy. With 4-way it's better for few-CTA cases. I'll do the 4-way with an inner unrolled loop. Actually — careful with the inner `if (idx < N) yr[idx] = ...`: for most iterations all 4 ifs true; branch predication handles. The guard `idx < N` depends on threadIdx/b/N ✓.

Hmm, wait — actually, should pass 2 reload x from global? Yes (no space to cache). Traffic: read + write. ✓.

Alternatively, could skip re-reading x in pass 2 if... no, need values. OK.

Let me now also double check "no dynamic shared memory, no cp.async, no cooperative groups" — we use static shared (allowed), plain shuffles (allowed — they're not cooperative groups). ✓. "No external libraries" — only common.cuh and CUDA built-ins. ✓. Do I need math constants: -FLT_MAX from <cfloat> or use __int_as_float(0xff7fffff)? Including <cfloat>... "no external libraries" — standard C++ headers are fine presumably (common.cuh likely includes things). Safer to avoid extra includes and define constants via literals/float.h? Hmm, common.cuh exists and defines EXP; it may include stuff. I'll avoid including anything except "common.cuh" and define:

```cuda
#define NEG_FLT_MAX (-3.402823466e+38f)
#define NEG_INF (-__int_as_float(0x7f800000))  // hmm
```
-INFINITY: could write (-1.f/0.f)? That's a division by zero constant → compiler folds to -inf. Or `__int_as_float(0xff800000)` = -INF bit pattern. I'll use __int_as_float(0xff800000) for -INF and -FLT_MAX via literal -3.4028234663852886e+38f or __int_as_float(0xff7fffff). Clean: 

```cuda
__device__ __forceinline__ float ninf() { return __int_as_float(0xff800000); }
```
Or just constants: `const float NEG_INF = __int_as_float(0xff800000);` inside kernel. And NEG_FLT_MAX = __int_as_float(0xff7fffff). These are exact. 

Hmm, do I even need -INF for padding? Padding value just needs to (a) not affect max (any very negative works: -FLT_MAX works IF... wait, issue identified earlier: padding = -FLT_MAX would contribute EXP(-FLT_MAX - nm) which is 0 unless nm = -FLT_MAX, which happens when chain max = -FLT_MAX... if padding = -FLT_MAX and real values also = -FLT_MAX: indistinguishable → sum gets extra +1s. Real -FLT_MAX inputs are absurd, but let's be exact: padding = -INF distinguishes properly (EXP(-INF - nm) = 0 for any nm > -INF... but if nm = -FLT_MAX: -INF - (-FLT_MAX) = -INF → EXP = 0 ✓). So padding -INF is strictly more correct. Use it.

Actually, hmm, wait: there's a subtlety with padding = -INF and chain init m = -FLT_MAX, s = 0: after processing padding element: nm = fmaxf(-FLT_MAX, -INF) = -FLT_MAX. s = 0 * EXP(-FLT_MAX - (-FLT_MAX)) + EXP(-INF - (-FLT_MAX)) = 0*EXP(0) + EXP(-inf) = 0 + 0 = 0 ✓. m stays -FLT_MAX ✓. Good.

Now, alternative approach for pass 1 without padding: structure loops so each chain has its own loop with its own bound:

```cuda
#pragma unroll
for (int k = 0; k < 4; k++) {
    for (int idx = tid + k*128; idx < N; idx += 512) { ... }
}
```
No — that serializes chains (k-loop outer), killing ILP... actually the compiler could interleave? The k-loop with #pragma unroll: 4 independent inner loops; compiler may software-pipeline/interleave them. Not guaranteed. The base-loop with 4 guarded loads is explicit ILP. Keep that. The guards `idx < N` inside: when idx ≥ N use -INF.

Hmm, but actually, hmm, one more thought: is 4-way the right width? For N < 512, the loop runs ≤ 1 iteration with some chains padded — fine, no waste (guards skip loads). For huge N, 4 chains give good ILP. OK.

Let me also reconsider: should I use __ldg explicitly for loads? `const float* __restrict__` → nvcc generates LDG.E.CONSTANT? With __restrict__, the compiler can use non-coherent load path. To be explicit and safe: `__ldg(&xr[idx])`. __ldg is a builtin, not an external library ✓. I'll use __ldg for the x loads (both passes). It returns float for float*. ✓.

Now let me also think about whether I should compute `EXP(v - nm)` — note when v ≤ m (common case: element not new max): nm = m; EXP(m - nm) = EXP(0) = 1; s += EXP(v - m). Two EXPs. Could optimize with branch `if (v > m)` — DATA-DEPENDENT control flow — forbidden! So keep branchless 2 EXPs. ✓ (The requirement explicitly says use fmaxf, not if — so branchless it is.)

Now, the reduction helpers. Let me write them carefully.

```cuda
__inline__ __device__ float warp_reduce_max(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__inline__ __device__ float warp_reduce_sum(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
```

These are fine. fmaxf is a CUDA math builtin ✓ (not an external library; it's in device runtime). The requirement says use fmaxf ✓.

Kernel:

```cuda
extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    constexpr int BLOCK = 128;           // blockDim.x is fixed at 128
    constexpr int NWARP = BLOCK / 32;    // 4
    constexpr int STRIDE = BLOCK * 4;    // 512

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    const size_t row = blockIdx.x;
    const float* __restrict__ xr = x + row * (size_t)N;
    float* __restrict__ yr = y + row * (size_t)N;

    __shared__ float s_max[NWARP];
    __shared__ float s_sum[NWARP];

    const float NEG_INF = __int_as_float(0xff800000);
    const float NEG_FLT_MAX = __int_as_float(0xff7fffff);

    // ---- Pass 1: online max + sum ----
    float m0 = NEG_FLT_MAX, m1 = NEG_FLT_MAX, m2 = NEG_FLT_MAX, m3 = NEG_FLT_MAX;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

    for (int base = tid; base < N; base += STRIDE) {
        const int i0 = base + 0 * BLOCK;
        const int i1 = base + 1 * BLOCK;
        const int i2 = base + 2 * BLOCK;
        const int i3 = base + 3 * BLOCK;

        const float v0 = (i0 < N) ? __ldg(xr + i0) : NEG_INF;
        const float v1 = (i1 < N) ? __ldg(xr + i1) : NEG_INF;
        const float v2 = (i2 < N) ? __ldg(xr + i2) : NEG_INF;
        const float v3 = (i3 < N) ? __ldg(xr + i3) : NEG_INF;

        float nm;
        nm = fmaxf(m0, v0); s0 = s0 * EXP(m0 - nm) + EXP(v0 - nm); m0 = nm;
        nm = fmaxf(m1, v1); s1 = s1 * EXP(m1 - nm) + EXP(v1 - nm); m1 = nm;
        nm = fmaxf(m2, v2); s2 = s2 * EXP(m2 - nm) + EXP(v2 - nm); m2 = nm;
        nm = fmaxf(m3, v3); s3 = s3 * EXP(m3 - nm) + EXP(v3 - nm); m3 = nm;
    }

    // combine the 4 per-thread streams
    float lm = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
    float ls = s0 * EXP(m0 - lm) + s1 * EXP(m1 - lm)
             + s2 * EXP(m2 - lm) + s3 * EXP(m3 - lm);

    // ---- block reductions ----
    lm = warp_reduce_max(lm);
    if (lane == 0) s_max[wid] = lm;
    __syncthreads();
    const float M = fmaxf(fmaxf(s_max[0], s_max[1]), fmaxf(s_max[2], s_max[3]));

    float ps = ls * EXP(lm... 
```

WAIT. Bug: I overwrote lm with warp_reduce_max(lm) — the full-warp-reduced max (same for all lanes in the warp). But I need the ORIGINAL per-thread lm for scaling ls! Let me keep separate variables:

```cuda
    float lm = ...;   // per-thread max (relative to which ls is computed)
    float ls = ...;   // per-thread sum relative to lm

    float wm = warp_reduce_max(lm);          // warp-level max (all lanes have it)
    if (lane == 0) s_max[wid] = wm;
    __syncthreads();
    const float M = fmaxf(fmaxf(s_max[0], s_max[1]), fmaxf(s_max[2], s_max[3]));

    float cs = ls * EXP(lm - M);             // rescale to global max
    cs = warp_reduce_sum(cs);
    if (lane == 0) s_sum[wid] = cs;
    __syncthreads();
    const float S = s_sum[0] + s_sum[1] + s_sum[2] + s_sum[3];
```

Check the rescale edge again: lm (per-thread) = -FLT_MAX, M finite: EXP(-FLT_MAX - M) → hmm: -FLT_MAX - M where M is finite, say M = 5: -3.4e38 - 5 = -3.4e38 (rounded) → EXP(-3.4e38) = 0 → cs = 0 * 0 = 0 ✓ (ls = 0 too). If M = -FLT_MAX (row max = -FLT_MAX): threads with lm = -FLT_MAX: EXP(0) = 1 → cs = ls * 1; ls = Σ EXP(v + FLT_MAX)... v ≤ -FLT_MAX... if v = -FLT_MAX: EXP(0) = 1 → contributes. Threads' ls values consistent ✓. S = count of -FLT_MAX elements (+0 from -INF elements) ≥ 1 if any element is -FLT_MAX. y = EXP(v - M) * (1/S) = 1/S each ✓ correct softmax! Nice, even that degenerate case works. Only ALL-(-INF) row breaks (S=0). And actually wait, ALL-(-INF) rows: every softmax impl outputs NaN/undefined; fine.

Hmm, one more: N ≥ 1 guaranteed, so M is the max of at least one real element... M = max over threads of lm; threads with elements have lm ≥ their elements... M ≥ max element ≥ -FLT_MAX... if the only element is -INF: M = -FLT_MAX? Thread's lm: chain m0 init -FLT_MAX, v0 = -INF: nm = fmaxf(-FLT_MAX, -INF) = -FLT_MAX → lm = -FLT_MAX ✓ (as computed). S = 0 → NaN. OK degenerate.

Now S: computed identically by all threads (reads s_sum after sync) ✓. M: all threads compute from s_max ✓.

Pass 2:

```cuda
    const float inv = 1.0f / S;

    for (int base = tid; base < N; base += STRIDE) {
        const int i0 = base + 0 * BLOCK;
        const int i1 = base + 1 * BLOCK;
        const int i2 = base + 2 * BLOCK;
        const int i3 = base + 3 * BLOCK;

        if (i0 < N) yr[i0] = EXP(__ldg(xr + i0) - M) * inv;
        if (i1 < N) yr[i1] = EXP(__ldg(xr + i1) - M) * inv;
        if (i2 < N) yr[i2] = EXP(__ldg(xr + i2) - M) * inv;
        if (i3 < N) yr[i3] = EXP(__ldg(xr + i3) - M) * inv;
    }
```

Good. Now — think about whether the `if (i0 < N)` in pass 2 and ternaries in pass 1 satisfy "control flow depends only on threadIdx/blockIdx/blockDim/N": i_k = base + k*BLOCK where base = tid + j*STRIDE → depends on threadIdx and loop counter (which depends on N via bound) and N ✓. No data-dependent branching ✓. All branches are index-bounds checks, uniform-per-warp mostly, and the verifier should accept them since conditions are functions of threadIdx/N.

Hmm, wait — but the loop condition `base < N` itself: depends on tid and N ✓ fine.

One thing to double-check: the verifier statement "Control flow may depend only on threadIdx/blockIdx/blockDim/N, never on data values (use fmaxf(a, b), not if (a > b))". Our code: `if (i0 < N)` — i0 derives from threadIdx + constants, N is N ✓. `if (lane == 0)` ✓ threadIdx. Loop `base < N` ✓. `for (int off = 16; off; off >>= 1)` ✓ compile-time. No other branches. ✓✓

Also "Write every exponential as EXP(v)" — every exp appears as EXP(...) ✓. We have EXP in: pass1 8 calls, combine 4, rescale 1, pass2 4. All via macro ✓. No expf/__expf anywhere ✓. But note: does anything else implicitly use exp? `powf`? No. `1.0f / S` — division, not exp ✓.

Include "common.cuh" — must be first. The macro EXP is defined there. I must NOT define EXP myself. ✓.

Now — "no dynamic shared memory": we use static __shared__ arrays ✓. "no cp.async" ✓ none. "no cooperative groups" ✓ none (raw shuffles fine). "no external libraries" — we use only CUDA intrinsics: __shfl_xor_sync, __ldg, __int_as_float, fmaxf — all device built-ins, fine.

Launch config: grid = rows, blockDim.x = 128 — that's the caller's job; our kernel just must assume blockDim.x == 128. I hard-code BLOCK=128 but could also use blockDim.x... The spec says blockDim.x = 128 exactly, so I can rely on it. Using constexpr 128 is fine and enables compile-time optimization. Hmm, but to be safe if launched differently? The spec guarantees 128. I'll use constexpr but maybe guard: `const int BLOCK = blockDim.x;`? If I use blockDim.x directly, STRIDE = 4*blockDim.x is runtime — fine too. But constexpr unrolls better. The requirement is explicit: blockDim.x = 128 exactly. Use constexpr 128. Hmm, but what if the harness launches with 128 as promised — yes. I'll add `assert`? No asserts (would need include, and assert is control flow on data? assert(blockDim.x==128) — threadIdx/blockDim-dependent, allowed, but let's not; keep clean). Actually, let me make the code work for blockDim.x being a multiple of 32... no, keep it simple: assume 128, per spec. Actually, you know what — let me write it to use blockDim.x where natural but with the stride computed from it... No: warp count NWARP = 128/32 = 4 hardcoded for the shared arrays. If blockDim were different, breaks. Spec says 128 exactly. Fine.

Hmm, wait, actually let me reconsider using blockDim.x in the code so it's robust: 
- s_max[4] assumes ≤ 4 warps... could size [32] (max warps) — then NWARP = blockDim.x/32 runtime. The final combine `fmaxf(fmaxf(s_max[0],s_max[1]), ...)` assumes exactly 4. I'd need a loop over warps: `float M = -FLT_MAX; for (int w = 0; w < NWARP; w++) M = fmaxf(M, s_max[w]);` — loop bound depends on blockDim ✓ allowed. That's more robust. But the spec fixes 128; hardcoding lets me unroll and is faster. I'll hardcode 128 with a comment. It matches "blockDim.x = 128 exactly".

Now, performance details:

1. The pass-1 loop: 4 loads (LDG) → 4×(FMNMX, FADD, EXP, FMA...) — the compiler will schedule loads together (they're independent) → good MLP (memory-level parallelism): 4 in-flight loads per thread → per warp 4×128B = 512B in flight → per SM (64 warps) 32KB in flight → 108 SMs → 3.4MB in flight → vs ~700KB needed to saturate DRAM ✓ great.

2. Pass-2 loop: 4 loads + 4 stores; stores depend on loads+EXP. The loads issue together ✓.

3. Register usage: m0..m3, s0..s3 (8), v0..v3 (4), base/indices, pointers, plus reduction temps — maybe 32-40 registers → 16 CTAs/SM? 128 threads × 40 regs = 5120 regs/block... A100 has 65536 regs/SM → 12.8 blocks → 12 blocks/SM × 128 = 1536 threads = 48 warps — decent (max 64). Fine.

4. __syncthreads ×2 per row + shuffles — for small N (many rows), overhead per row: ~2 syncs (each ~ tens of cycles) + 10 shuffle rounds (5 max + 5 sum ≈ each ~5-10 cycles latency... warp reduce = 5 shuffles, each ~10-20 cycles? __shfl has ~10 cycle latency? Typically ~5 shuffles × ~15 cycles ≈ 75 cycles per reduce, two reduces + 2 syncs ≈ 250-400 cycles per row. For N=128 (one element per thread): work per row ≈ load 512B + ~100-arith + reduction ≈ maybe 600-800 cycles → ~... with 16 CTAs/SM → fine, it's the standard cost. Could reduce syncs by single-warp-per-row designs, but grid/block are fixed by spec. This is the right design given constraints.

Alternatively for tiny N (N ≤ 128), could skip pass 1's strided loop complexities... no, it's already minimal: loop executes ≤ 1 iteration with guards. Fine.

5. Coalescing ✓ (128 consecutive floats per warp-load).

Let me now write the final combine of chains: `ls = s0*EXP(m0-lm) + ...` — 4 EXPs per thread once — negligible.

Also — hmm, one potential improvement: in pass 1, we could avoid the padding EXPs cost... negligible.

Another thought: should pass 1 use `__ldg`? x is const __restrict__, so loads are already LDG.E.CI (non-coherent)? nvcc does use LDG for const __restrict__ pointers with -arch flags? Historically, __restrict__ const enables __ldg-style codegen... To be sure, explicit __ldg. ✓.

Now — one more correctness check on the online update with padding: consider N = 130. Thread 0: i0 = 0 ✓, i1 = 128 ✓ (128 < 130), i2 = 256 ✗ pad, i3 = 384 ✗ pad. Thread 1: i0 = 1 ✓, i1 = 129 ✓ (129 < 130), i2 ✗, i3 ✗. Threads 2-127: i0 = tid ✓ (≤127 < 130 ✓), i1 = tid+128 ≥ 130 ✗ pad. So elements 0..127 by i0, 128 by thread0-i1, 129 by thread1-i1 ✓ all 130 elements exactly once ✓. Loop: base=tid < 130 → iterate; base += 512 → ≥ 130 exit ✓.

Chain assignment: element 128 → chain 1 of thread 0; element 129 → chain 1 of thread 1. ✓ covered by the max/sum combine.

Edge N = 1: thread 0 loads x[0]; all others pad (their i0 = tid ≥ 1 → pad -INF... wait: i0 = base + 0 = tid; for tid ≥ 1: tid < N=1 false → pad ✓). Chains: thread 0: m0 = x[0], s0 = 1 (if x[0] finite: EXP(x0 - x0) = 1) — wait: v0 = x[0], nm = fmaxf(-FLT_MAX, x0) = x0; s0 = 0*EXP(-FLT_MAX - x0) + EXP(0) = 0 + 1 = 1 ✓. Others: all chains -FLT_MAX/0. lm(thread0) = x0, ls = ... combine: m = x0; ls = s0*EXP(0) + s1*EXP(-FLT_MAX - x0)*0... careful: s1 = 0, m1 = -FLT_MAX: term = 0 * EXP(-FLT_MAX - x0) = 0 * 0 = 0 ✓ (EXP(-FLT_MAX - x0): if x0 = 1e30: -3.4e38 - 1e30 → -inf? -3.4e38 - 1e30 = -3.5e38 → overflows to -INF → EXP(-INF) = 0 ✓; if x0 = -1e30: -3.4e38 + 1e30 = -3.3e38 → EXP = 0 ✓). ls = 1 ✓. M = x0, S = 1, y[0] = EXP(0) * 1 = 1 ✓.

Edge: x values like 100 and -100: max = 100; EXP(-200) ≈ 1.38e-87 — no underflow issue (denormal at ~1e-45; e^-200 is way below denormal → 0). Fine — mathematically those terms are 0 in float anyway.

x ~ 100 uniform: max = ~100, EXP(0..small) fine.

Now, about `EXP` semantics: assume EXP(v) behaves like expf(v) or __expf(v) — either way our usage is standard: arguments are ≤ 0 always? Let's check: EXP(m0 - nm): m0 ≤ nm always (nm = fmaxf(m0, v0) ≥ m0) → arg ≤ 0 ✓. EXP(v0 - nm) ≤ 0 ✓. EXP(m_k - lm) ≤ 0 ✓. EXP(lm - M) ≤ 0 ✓. EXP(x - M) ≤ 0 ✓. All args ≤ 0 → results in (0, 1] → no overflow ever ✓. 

With __expf-style (EX2-based, ~2 ulp): fine.

Now let me reconsider the two __syncthreads and shared arrays once more for races:

Sequence per thread:
1. lm/ls computed (registers).
2. wm = warp_reduce_max(lm) — shuffles, all lanes get warp max.
3. if (lane==0) s_max[wid] = wm; — 4 writes (one per warp).
4. __syncthreads(); — all writes visible.
5. M = fmaxf over s_max[0..3] — all threads read shared (after sync) ✓.
6. cs = ls * EXP(lm - M) — registers.
7. cs2 = warp_reduce_sum(cs).
8. if (lane==0) s_sum[wid] = cs2; — writes to DIFFERENT array. Any thread still reading s_max at step 5 concurrently with step 8 writes to s_sum — no conflict ✓.
9. __syncthreads();
10. S = s_sum[0..3] sum ✓.

✓ Correct, 2 syncs. 

Then inv = 1/S: every thread computes the same division (or use __frcp? plain `/` is IEEE-correct-ish; fine).

Pass 2 loop as above.

Let me now also think about whether pass 2 should recompute... it reads xr again. Note x may alias y? Signature has separate in/out; assume no alias (and __restrict__ promises no alias).

One more think on possible N values up to INT_MAX: loop counters int — base < N with base up to ~N+512 — int fine (N ≤ 2^31-1, base ≤ N + 

## content

