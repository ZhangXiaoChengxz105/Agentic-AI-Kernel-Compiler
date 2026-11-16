# tirx round 1

model=glm-5.3 in=7736 out=18352 cost=0.08819 latency=107.4s finish=stop effort=high usage={"completion_tokens": 18352, "completion_tokens_details": null, "prompt_tokens": 7736, "prompt_tokens_details": null, "reasoning_tokens": 16881, "total_tokens": 26088}

## reasoning
Let me design the kernel. rows, n static. grid=rows, block=128 (4 warps). One CTA per row.

Plan:
1. Each thread reads strided elements of row r, tracks local max.
2. Warp-reduce max with shfl butterfly, warp leaders store to smem, warp 0 reduces, broadcast via smem.
3. Same for sum of exp(x - max).
4. Rescale and store.

Use vectorized loads where possible? For performance, use v4 loads (4 floats = ld.global.v4.b32) when n % 4 == 0? But requirement: correct for ANY n >= 1. Simplicity: use scalar loads with stride 128; can do two loads per iteration (strided by 128 within a tile of 256?) Actually simple approach: each thread processes elements col = tid + k*128 for k in ceil(n/128). That's fine; loads are coalesced within a warp. For better ILP, process chunks of 4*128: each thread loads 4 consecutive elements (float4) at offset k*512 + tid*4 — that's coalesced float4 loads. But guard tail. Could branch: if n % 4 == 0, use vectorized path; else scalar path. Since n is static (Python int), we can choose at trace time. For high performance aim, implement vector path when n % 4 == 0, plus scalar fallback otherwise.

Also could cache x in shared memory or registers to avoid second global read. Option: keep in registers via local arrays sized niter*4 floats — for large n that's too many registers. Alternative: re-read from global (L2/L1 cached) — typical softmax kernels re-read or use smem. With block=128 and 4 floats per thread per iteration, for moderate n registers could hold values if we bound iterations. Simpler: store x values in shared memory (float32 smem of size n) so second pass reads from smem — that avoids second global load. n could be large though (smem 48KB default → n up to 12288 fits without dynamic smem opt-in; smem_pool may handle allocation). Actually smem_pool alloc — does it request dynamic shared memory automatically? Probably yes. But for very large n (>48KB... A100 has 164KB max smem per CTA). To be safe for ANY n, don't stage in smem; just re-read global (L2 will hit). Many production softmax kernels re-read. Alternatively: only stage in smem if n <= some threshold, else re-read global. That adds complexity but fine since trace-time Python decides.

Let me keep it reasonably simple but performant:

- Vectorized path when n % 4 == 0: each thread loads float4 chunks.
- Otherwise scalar path.
- Two passes over data (max, then sum), third pass writes output — actually second pass can compute sum and store exp values... but exp(x-max) needed again for output. Either store e = exp(x - max) into y during sum pass, then third pass divides y by sum — that's a y read+write in pass 3, same cost as re-reading x. Alternatively stage x or exp in smem when fits.

Simplest performant: 
Pass 1: read x (global), local max → block reduce max.
Pass 2: read x again (global, L2/L1 hit), compute e = exp(x - m), local sum → block reduce sum.
Pass 3: read x again or recompute... Hmm pass 3 needs exp again. Options: pass 2 writes e to y (global), pass 3 reads y and multiplies by 1/sum and writes back. That's 2 global writes + 1 extra read of y. Or pass 2 stores e in smem if n fits, then pass 3 reads smem, scales, writes y.

Smem staging: if n * 4 bytes <= ~48KB (or 96KB?), stage x in smem in pass 1 (store to smem as loaded), pass 2 reads smem, computes exp, accumulates sum, and writes e to y global. Pass 3 reads y, multiplies by inv sum, writes y. Still needs third pass over y.

Alternative: two passes total if we recompute exp in pass 3 from smem x: pass 1 load x → smem + max reduce; pass 2 read smem, compute exp, sum reduce; pass 3 read smem, exp again (redundant compute, cheap: exp is a few FLOPs / MUFU) scale and store. exp via __expf ≈ ex2.approx. That's fine: 2 global reads→1, 1 global write, smem reads.

But for n not fitting smem, fall back: re-read global.

Since n is static, choose strategy at trace time:
- if n <= SMEM_ELEMS (say 12288 = 48KB): stage x in smem, 1 global read.
- else: re-read global each pass.

Actually even simpler and still good: always stage in smem if n*4 <= 100KB? A100 max dynamic smem per block is 163KB (164KB - 1KB). Default without opt-in attribute is 48KB. Does txl handle dynamic smem >48KB automatically? Unknown — safer to cap at 48KB.

Actually, hmm — do we even need smem staging? Reading global twice is fine performance-wise (L2 hit). Keep kernel simpler: pass 1 read global compute max; pass 2 read global compute exp & sum; pass 3 read global, exp, scale, write. That's 3 global reads (2 from cache). For n up to large, fine.

But "aim for high performance" — let's do the smem staging for n <= 12288 and global reread otherwise, with vectorization when n%4==0. Hmm, that's 2 (staging) × 2 (vector) = 4 code paths. Maybe simplify: use scalar loads always (stride pattern tid + k*128). Coalesced 128B per warp per load — that's fully efficient for global throughput actually (each warp loads 32 consecutive floats = 128B). The main loss vs float4 is instruction count and fewer outstanding loads. For a memory-bound softmax, scalar coalesced loads achieve near peak. I'll do scalar for simplicity but add per-thread unrolling by 4 with independent loads (offsets tid, tid+128, tid+256, tid+384 per iteration of 512) to improve ILP. That works for any n with guards.

Actually let me reconsider: guarded scalar loads inside loop with per-element `if` is fine. Let me write:

UNROLL = 4 elements per thread per iteration? For small n (<128) many threads idle. Fine.

Let me structure:

```python
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF

def make_kernel(rows: int, n: int):
    BLOCK = 128
    NTHREADS = BLOCK

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        NW = BLOCK // 32  # 4

        smem = txl.smem_pool()
        red = smem.alloc([NW], "float32")

        v = txl.local_scalar("float32")
        m = txl.local_scalar("float32")
        s = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")

        def warp_reduce_max(val):
            peer = txl.local_scalar("float32")  # hmm, local per call - fine, it's a traced local
            ...
```

Careful: helpers creating txl.local_scalar each call — since helpers are called multiple times, each call creates new local. That's fine.

Warp max via shfl butterfly:

```python
def shfl_bfly(src_f32, delta):
    peer = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(src_f32, ty="uint32"),
                               txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
    return txl.reinterpret(peer, ty="float32")
```

Wait — shfl_sync.bfly.b32 signature from example: `txl.ptx.shfl_sync.bfly.b32(dst, src, delta, clamp, mask)`. Good.

For max reduction: compare and select — use txl.Select(cond, a, b)? Or use fmax? PTX has max.f32: `txl.ptx.max.f32(dst, a, b)`. Does txl.ptx have max.f32? txl.ptx is generic PTX spelling — `txl.ptx.max.f32(d, a, b)` should work since ld/st/mov/bar/shfl all work that way. I'll use it; it's standard PTX. Alternatively use txl.Select on a>b. To be safe, use txl.max? There might be txl.max intrinsic. The RMSNorm example used txl.rsqrt — so math intrinsics exist under txl (rsqrt, probably exp too). Let me use txl.max(a, b) — hmm, not verified. txl.rsqrt was used; likely txl.exp, txl.log exist too. Risky. Safer: txl.Select? Select returns value. Actually there's `txl.Select` shown in README. Use:

```python
m = txl.Select(a > b, a, b)
```

But Select is likely a TIR node producing a select — fine.

Alternatively use ptx max.f32 which I'm fairly confident exists since it's plain PTX scalar op like add/mul. The guide says "Plain f32 scalar ops" are fine. I'll use `txl.ptx.max.f32(dst, a, b)` and `txl.ptx.add.f32`. Hmm, actually there might be txl idioms. The scale2 example used Python `+` and `*` on traced values (v * txl.float32(2.0)). So arithmetic operators work. For max, use txl.Select(a > b, a, b). Comparisons: `col < n` used, so > works.

For exp: does txl.exp exist? txl.rsqrt exists in the RMSNorm example. Likely txl.exp exists as an intrinsic mapping to __expf or exp? Precision: we need decent accuracy. `exp` in TIR typically maps to libdevice exp (accurate) — fine. Numerical stability: x - max ∈ [-inf, 0], exp of that fine. For x ~ -100 relative to max → exp → 0, fine.

To be safe against txl.exp not existing, could use PTX ex2.approx.f32: exp(x) = ex2(x * log2e). `txl.ptx.ex2.approx.f32(dst, src)` — PTX instruction name with dots; txl.ptx[LD_G_V4] pattern shows arbitrary instruction names via getitem, and `txl.ptx.st.global_.f32` attribute access. Attribute access with dots: `txl.ptx.ex2.approx.f32` — attribute chaining... `txl.ptx.st.global_.f32` works, so `txl.ptx.ex2.approx.f32` likely works too. But precision of ex2.approx: max error ~2 ulp, fine for softmax. However, the interface might not support attribute names with multiple dots? "global_" shows dot in name is escaped via underscore. "ld.global.v4.b32" used getitem. So I can use txl.ptx["ex2.approx.f32"](dst, src). Hmm, but does the DSL accept that instruction? It emits raw PTX. I think safest is txl.exp if it exists... I can't verify. The RMSNorm example used txl.rsqrt — intrinsics under txl namespace. Common set: sqrt, rsqrt, exp, log, pow, sin, cos, floor, ceil, fabs, min, max. I'd bet txl.exp and txl.max exist. But "never invent APIs" — hmm. I need to pick. 

Given txl.rsqrt confirmed, txl.exp is highly likely (TIR intrinsics exp, expf, exp2...). Also txl.max likely maps to TIR max intrinsic. I'll use txl.ptx for max (plain ptx max.f32 is definitely valid PTX and txl.ptx forwards arbitrary PTX spellings) and for exp use txl.exp. Hmm, mixed risk.

Actually let me think about what's safest overall. txl.ptx is described as "PTX and CUDA instructions are spelled through txl.ptx". Arbitrary PTX instructions like `ex2.approx.f32`, `max.f32`, `add.f32` should be spellable. The DPS form: destination first. For max.f32: `txl.ptx.max.f32(dst, a, b)` — attribute access "max" then ".f32" — `txl.ptx.max.f32` parses as attribute chain: txl.ptx . max . f32. The `st.global_.f32` example shows the object supports nested attribute access returning an instruction callable where dots become... "global_" has underscore for the dot? Actually "st.global_.f32" — the PTX is `st.global.f32`? Hmm, they wrote `global_` maybe because `global` is a Python keyword. So `txl.ptx.st.global_.f32` = instruction "st.global.f32". And `txl.ptx.shfl_sync.bfly.b32` = "shfl_sync.bfly.b32". And `txl.ptx.mov.b32` = "mov.b32". So nested attributes join with dots. Then `txl.ptx.max.f32` = "max.f32" ✓. And `txl.ptx.ex2.approx.f32` = "ex2.approx.f32" ✓. And `txl.ptx.mul.f32`, `txl.ptx.add.f32`.

But do these non-void instructions work in DPS form with a destination local? The shfl example: yes, `txl.ptx.shfl_sync.bfly.b32(peer, ...)` with peer a local_scalar — non-void with DPS. Good. So I can implement everything in PTX-level ops.

For exp accuracy: use ex2.approx.f32 after multiplying by log2(e)? __expf does exactly that. Error negligible for softmax. Alternatively use precise exp via txl.exp. I'll use txl.exp — hmm. Decision: I'll use ex2.approx.f32 via ptx since it's guaranteed spellable and fast. Numerically: exp(x-max) where x-max in [-inf, 0]. ex2.approx has ~2 ulp relative error; final softmax error ~1e-7 relative. Fine.

Actually wait — for large |x| ~ 100, x - max could be like -200, exp(-200) ≈ 1e-87, denormal-ish but fine in f32 (min normal ~1e-38; 1e-87 flushes to 0 with ftz or denormal). Sum still correct.

Plan details:

NW = 4 warps.

Reduction helper for max and sum:

```python
def warp_bfly_uint32(dst, src_u32, delta):
    txl.ptx.shfl_sync.bfly.b32(dst, src_u32, txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF))
```

Butterfly with XOR semantics: shfl.sync.bfly.b32 uses laneid ^ delta. Wait, actually PTX shfl.bfly mode: "the source thread index is the caller's lane ID XOR'd with the delta". So deltas 16,8,4,2,1 give full reduction. Good.

Max reduce warp:

```python
def warp_reduce_max(val):  # val is a local_scalar ref or expression
    cur = txl.local_scalar("float32")
    txl.assign(cur, val)
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(cur, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
        pv = txl.reinterpret(peer, ty="float32")
        txl.assign(cur, txl.Select(cur > pv, cur, pv))
    return cur
```

Hmm, but with inactive threads (n < 128), threads with no elements must contribute -inf (or a very small number) to the max. Initialize local max = -3.4e38 (or -inf). Use -inf: txl.float32(float("-inf"))? Might work but safer to use a large negative finite like -3.402823466e38. Actually with max via Select, -FLT_MAX works since all real values > it. But careful: if a real value is exactly -FLT_MAX... edge, ignore, or use -inf via math.inf? txl.float32(-float("inf")) — constant folding to inf bits; Select(cur > pv) with NaN concerns none. I'll use -3.402823466e38.

Hmm, but Select vs ptx max.f32: max.f32 returns the non-NaN operand typically (PTX: if one is NaN, returns the other). No NaNs expected. I'll just use Select — no, actually simpler: I realize I could avoid Select by using `txl.ptx.max.f32(dst, a, b)`. Both fine. I'll use txl.Select since it's documented in README.

Sum reduce: init 0, add. Peer shuffle then add: `txl.assign(cur, cur + txl.reinterpret(peer, ty="float32"))` — the RMSNorm example used `+` on traced floats. Good.

CTA reduce:

```python
with txl.if_(lane == 0), txl.then_():
    txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
txl.ptx.bar.sync(0, txl.uint32(128))
```

Wait, bar.sync signature: `txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(nthreads))` from example. Good.

Then warp 0 reduces NW values:

```python
with txl.if_(warp == 0), txl.then_():
    val = txl.local_scalar("float32")
    with txl.if_(lane < NW), txl.then_():
        txl.ptx.ld.shared.f32(val, red.ptr_to([lane]))
    with txl.else_():
        txl.assign(val, txl.float32(0.0))  # for sum; for max, -FLT_MAX
    # warp reduce val
    ...
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.f32(red.ptr_to([0]), val)
txl.ptx.bar.sync(0, txl.uint32(128))
txl.ptx.ld.shared.f32(result, red.ptr_to([0]))
```

For max version, else branch sets -FLT_MAX. I'll parameterize the reduce helper with identity and combine function. Since combine differs (Select vs +), write two helpers or pass a python function.

Let me write a generic helper:

```python
def cta_reduce(val, ident, combine):
    cur = txl.local_scalar("float32")
    txl.assign(cur, val)
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(cur, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.assign(cur, combine(cur, txl.reinterpret(peer, ty="float32")))
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
    with txl.if_(warp == 0), txl.then_():
        w = txl.local_scalar("float32")
        with txl.if_(lane < NW), txl.then_():
            txl.ptx.ld.shared.f32(w, red.ptr_to([lane]))
        with txl.else_():
            txl.assign(w, ident)
        for delta in (16, 8, 4, 2, 1):
            txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(w, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
            txl.assign(w, combine(w, txl.reinterpret(peer, ty="float32")))
        with txl.if_(lane == 0), txl.then_():
            txl.ptx.st.shared.f32(red.ptr_to([0]), w)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
    out = txl.local_scalar("float32")
    txl.ptx.ld.shared.f32(out, red.ptr_to([0]))
    return out
```

Wait — issue: only warp 0 does the second reduction, but the first bar.sync must be executed by all threads. It is, since it's outside the if. But the second warp-reduce inside `warp == 0` — shfl_sync with FULL_MASK only in warp 0, all 32 lanes of warp 0 execute it (divergence only on lane<NW for the load; else branch sets ident; all lanes participate in shfl). Good.

But wait: there's a subtlety — the shfl in the first warp reduce (all warps, all lanes active since block=128 and no divergence at that point). Yes fine.

Also `red.ptr_to([lane])` — smem tensor ptr_to indexing; the example used `sum_sq_smem.ptr_to([warp])` where warp is an expression. Good.

Return: `out` is a local_scalar created in helper — returning the traced value is fine.

Barriers: need to be careful with reusing same barrier id and smem across passes — bar.sync before reuse is included. Between pass1's final read and pass2's write to red: pass 2's warp write to red[0..NW-1] — is there a race with pass1's reads of red[0]? Thread reads red[0] after bar.sync; then pass2 writes red after another shfl chain... Within same thread, read happens before write in program order, but another thread could write red[warp] before this thread read red[0]? Sequence: bar.sync #2 (after warp0 writes red[0]); all threads ld red[0]; then pass 2 computation (loads etc.); then lane0 of each warp st red[warp]. Is it guaranteed that all threads have read red[0] before any thread writes red? No! There's no barrier between the read of red[0] and the next write to red. Need a bar.sync before pass2's lane-0 stores, or use different smem slot / barrier. But note the warp reduce and the pass-2 loads happen between. Not guaranteed. Add a third bar.sync after the broadcast load, before the next reduction stores. Actually simplest: after ld red[0], add `txl.ptx.bar.sync(0, BLOCK)`. Since bar.sync with same id by all threads — fine.

Alternatively use two barriers alternately, or allocate red with NW+... simplest: add bar.sync after broadcast read. I'll add it.

Actually the RMSNorm example: after cta_sync(0), all threads ld sum_sq_smem[0], then compute, then in the next iteration's reduction they st sum_sq_smem[warp] after cta_sync(1)... wait they use bar 1 later. Hmm, they do `cta_sync(1)` at end of loop iteration and reductions use bar 0. Their stores to sum_sq_smem happen after loads, guarded by... warp_sum shfls etc. Actually in their code, the store `ST_S_F32(sum_sq_smem[warp])` in next iteration could race with the previous read of sum_sq_smem[0] by a slow thread? They have cta_sync(1) at loop end after pass 2 — which is after both reads and before next iteration's writes. OK.

I'll just add a bar.sync after each broadcast load. Cost negligible.

Main passes. Choose VEC path if n % 4 == 0 (load float4 via ld.global.v4.f32 → wait, PTX ld.global.v4.f32 exists; the example used ld.global.v4.b32 with uint32 destinations). For f32, could use ld.global.v4.f32 with float destinations — spellable? "ld.global.v4.f32" via txl.ptx["ld.global.v4.f32"]. Should be fine. But to reduce risk, use b32 with uint32 locals then reinterpret — the RMSNorm pattern. Actually ld.global.nc? Use plain ld.global (L1 caching fine).

Hmm, but do I want vectorization complexity? Performance: softmax is memory bound; scalar coalesced loads with 128 threads: each iteration warp loads 128B. With unrolling and multiple iterations, memory throughput can still reach peak since enough loads in flight (need enough ILP). With grid=rows CTAs and rows large, occupancy covers it. For simplicity and correctness for ANY n, scalar path only is acceptable, but let's add modest unroll: process 4 elements per thread per iteration with offsets tid + k*128, k = it*4 + j... Actually simple loop over ceil(n/128) iterations, each thread loads one element guarded. That's the scale2 pattern. Loads across iterations for a thread are strided 128*4 bytes — fine.

To improve ILP, I can unroll by 4 manually in Python: iterate over chunks of 512:

```python
nblk = (n + 511) // 512  # trace-time
with txl.serial(nblk) as it:
    base = it * 512
    for j in range(4):
        col = base + j * 128 + tid
        with txl.if_(col < n), txl.then_():
            load, update max
```

That gives 4 independent guarded loads per iteration → good ILP. Guards per element; fine.

Hmm wait — but for n not multiple, tail handled. Good. For n < 512, nblk=1, guards handle. For n=1: only thread 0 col 0 active. Correct: other threads have max=-FLT_MAX, sum=0; reductions handle. exp(x-max)=exp(0)=1, sum=1, y=1. Good.

Second pass: compute sum of exp(x - m). Third pass: y = exp(x - m) * inv. To avoid recomputing exp twice and reading x three times, stage x in smem when it fits:

```python
USE_SMEM = n * 4 <= 48 * 1024
```

If USE_SMEM: allocate smem x_smem [n] f32. Pass 1: load x global, store to smem, track max. bar. Then passes 2 and 3 read smem (fast). Pass 3 writes y global.

Wait, but pass 2 computes e = exp(x-m) and sums; pass 3 recomputes exp from smem x. Alternatively pass 2 stores e into smem (overwrite x_smem), pass 3 reads e, multiplies by inv, stores. Either way. Recompute exp is cheap (ex2 MUFU ~4 cycles throughput 1/4... fine). I'll recompute exp in pass 3 — no, storing e in pass2 avoids recompute; but overwriting x_smem while other threads read — need barrier between pass2 reads and writes. There's already a reduction barrier between. Sequence: pass2: read smem, compute e, sum local, store e to smem? Careful: threads read different elements than they write (same index actually: thread reads smem[col] and writes e to smem[col] — same col). Within a thread read-then-write same location is fine; other threads read their own cols. No cross-thread hazard. So pass 2 can overwrite in place safely. Then after reduction barrier, pass 3 reads e from smem, multiplies by inv (rcp of sum), stores to y.

inv = 1/sum: use precise division? txl operator `/` — the RMSNorm example used `ss[0] / txl.float32(hidden_size)`. So `/` works. Use division (TIR fdiv → div.rn probably). Fine.

If not USE_SMEM (large n): pass 2 and 3 re-read from global. Pass 3 recomputes exp. Write two code paths via Python if.

Actually, is smem staging worth the code complexity? Yes for n ≤ 12288 which is the common case, saves 2 global reads (well, they'd hit L2/L1). Let me include it.

Actually wait, one more consideration: smem staging requires barrier before pass 2 reads (all writes done). The reduction barriers already provide that (pass1 stores to smem happen before cta_reduce which starts with shfl then bar.sync — hmm, the smem x writes happen in pass 1 loop, then cta_reduce's first bar.sync ensures all smem writes visible before pass 2 reads. Yes: bar.sync in cta_reduce orders them. Good. But careful about `ld.shared` needing the data: bar.sync synchronizes and smem writes by all threads before bar are visible after. Yes.)

Similarly pass 2's e writes to smem: pass 2 loop writes e, then cta_reduce for sum has bar.sync before... wait, order in pass 2: read smem x, compute e, write e to smem, accumulate local sum, then cta_reduce(sum) which does shfl, then st red, bar.sync. The pass-3 reads of e happen after the final bar.sync of the sum reduction. But is there a bar between "thread A writes e[colA]" and "thread B reads e[colA]"? Pass 3 reads: each thread reads e at same cols it wrote? Pass 3: for each col assigned to this thread, read smem e[col], scale, store y. Thread reads the same cols it wrote in pass 2? Yes if pass 3 uses the same col assignment (same loop structure). Then no cross-thread dependency... but wait, thread reads e[col] which it wrote itself — program order guarantees visibility of its own writes? Within a thread, yes, its own writes to smem are visible to itself without a barrier. And other threads don't read those locations. So actually safe even without barrier. But the sum broadcast read requires the barrier anyway. Fine, safe.

Hmm, but there's a subtle issue: pass 3 reads red[0] broadcast after bar.sync #2 of the sum reduction. Then pass 3 (next kernel launch? no, same kernel, one row per CTA; done). No further reuse of red. OK.

Barrier IDs: use bar 0 throughout with proper sync between reuses. Let me trace barrier usage in cta_reduce:

1. bar.sync(0) [after warp leaders store red]
2. warp0 loads red, reduces, lane0 stores red[0]
3. bar.sync(0)
4. all threads load red[0]
5. (extra) bar.sync(0) to protect red reuse in the next cta_reduce call — needed before step 1's stores of next call. Add it.

But careful: the smem x writes in pass 1 happen before the first cta_reduce — visibility to pass 2 readers needs a bar between write and read: cta_reduce's bar at step 1 (or 3) provides it. Pass 2 reads happen after step 4's load, which is after step 3 bar. ✓.

Pass 2 writes e to smem; pass 3 reads e — same thread, ✓ (and reduction bars in between anyway).

Also: for the non-smem path, pass 2 and 3 re-read global x — fine.

Now, ld.shared/st.shared spellings: `txl.ptx.ld.shared.f32(dst, ptr)` and `txl.ptx.st.shared.f32(ptr, val)` — from example LD_S_F32 = "ld.shared.f32", used as txl.ptx[LD_S_F32](ss[0], sum_sq_smem.ptr_to([lane])). And st.shared.f32 similarly. Attribute form `txl.ptx.ld.shared.f32` should also work but "shared" is fine as attribute name. I'll use attribute access.

Global: `txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))` — note "global_" due to keyword. And st.global_.f32.

Wait — x.ptr_to([r, col]): r is cta_id (int32?), col int32. Tensor shape (rows, n) with Python ints — static shape, indices are runtime expressions. Fine.

exp: implement helper:

```python
LOG2E = 1.4426950408889634

def expf(dst, val):
    txl.ptx.ex2.approx.f32(dst, val * txl.float32(LOG2E))
```

Hmm, `txl.ptx.ex2.approx.f32` — attribute chain: ex2 . approx . f32. OK. Alternatively use `txl.ptx["ex2.approx.f32"]`. I'll use the getitem form to be safe with names? The example used both attribute (`txl.ptx.mov.b32`) and getitem (`txl.ptx[LD_G_V4]`) forms. Both fine. I'll use attribute form for ex2.

Hmm wait, is ex2.approx precise enough for "numerically stable"? Yes: softmax with ex2.approx gives ~1e-6 relative error, standard practice (__expf). The problem says stable for large |x| ~ 100, which refers to overflow avoidance (subtract max). Fine.

Actually, should I worry that exp argument x - m can be very negative → ex2.approx.f32 of very negative → 0. Fine. What about m = -FLT_MAX when... no, m is actual max of row.

Edge: NaN inputs — undefined behavior acceptable.

Precision concern: sum could accumulate error; use f32 sum, fine.

Now, division: y = e * (1.0 / s). Use `txl.ptx.rcp.approx.f32`? Or precise division. Approx rcp has 1 ulp-ish error (2^-23ish? rcp.approx max error 2^-23? Actually rcp.approx.f32 has max relative error 2^-23... no, it's about 2^-23? The documented max ulp error for rcp.approx is 1 ulp? For rcp.approx.f32: maximum ulp error = 1... hmm, actually it's 2^-23 relative? Either way ~1e-7, fine). But safer numerically to use exact division: `/` operator on traced floats → div.rn.f32 presumably (TIR fdiv with default rounding). The RMSNorm used `/`. I'll compute inv = txl.float32(1.0) / s_sum then multiply. Division once per thread; cost negligible. Use `/`.

Hmm, does `/` on traced values produce fdiv? Presumably. OK.

Now write the code.

Vectorization consideration: skip; scalar loads with 4x unroll. Actually wait — with unroll-by-4 over 512-element tiles, for each j we do a guarded `if col < n`. For n%512 tails this is fine.

But actually there's a subtle perf point: with USE_SMEM and unrolled loads, we do: ld.global (guarded) → st.shared. Good.

Let me also consider: for n <= 128 (small), many threads idle but correctness fine.

Let me now also handle the case where n is large and not smem: same code but pass 2/3 load from global instead of smem. Write as Python-level branching with helper closures.

Structure:

```python
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
NEG_HUGE = -3.4028234663852886e38
LOG2E = 1.4426950408889634

def make_kernel(rows: int, n: int):
    BLOCK = 128
    NWARP = BLOCK // 32
    UNROLL = 4
    TILE = BLOCK * UNROLL
    NTILE = (n + TILE - 1) // TILE
    USE_SMEM = (n * 4) <= 48 * 1024

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        red = smem.alloc([NWARP], "float32")
        if USE_SMEM:
            xs = smem.alloc([n], "float32")

        m = txl.local_scalar("float32")
        s = txl.local_scalar("float32")
        v = txl.local_scalar("float32")
        e = txl.local_scalar("float32")

        def shfl(cur):
            peer = txl.local_scalar("uint32")
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer, txl.reinterpret(cur, ty="uint32"),
                    txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
                txl.assign(cur, cur + txl.reinterpret(peer, ty="float32"))
```

Hmm wait — for the max reduction I need a different combine. Let me write two separate helpers or pass combine function:

```python
        def cta_reduce(val, ident, combine):
            cur = txl.local_scalar("float32")
            txl.assign(cur, val)
            peer = txl.local_scalar("uint32")

            def warp_red(acc):
                for delta in (16, 8, 4, 2, 1):
                    txl.ptx.shfl_sync.bfly.b32(
                        peer, txl.reinterpret(acc, ty="uint32"),
                        txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
                    txl.assign(acc, combine(acc, txl.reinterpret(peer, ty="float32")))

            warp_red(cur)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            with txl.if_(warp == 0), txl.then_():
                w = txl.local_scalar("float32")
                with txl.if_(lane < NWARP), txl.then_():
                    txl.ptx.ld.shared.f32(w, red.ptr_to([lane]))
                with txl.else_():
                    txl.assign(w, ident)
                warp_red(w)
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx.st.shared.f32(red.ptr_to([0]), w)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            res = txl.local_scalar("float32")
            txl.ptx.ld.shared.f32(res, red.ptr_to([0]))
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            return res
```

Issue: `peer` created once in cta_reduce, used in nested warp_red — fine. But `warp_red(w)` where w is local_scalar — assign into it. OK. Also `txl.assign(acc, combine(...))` — acc is a local_scalar, fine.

Wait, a subtle issue with the second bar.sync (step 3 above) and warp0's store to red[0]: warp0 stores red[0] then bar.sync; other warps are already at bar? All threads reach the same bar.sync(0). Threads in warp0 diverge into the if but reconverge before bar. ✓.

Third bar.sync protects red reuse. But note: it also must not deadlock — all threads execute it. ✓.

Hmm, one more subtlety: the first warp_red inside `if warp==0` — wait no, first warp_red is outside, all warps. ✓.

combine functions:

```python
        def fmax(a, b):
            return txl.Select(a > b, a, b)
        def add(a, b):
            return a + b
```

Hmm, txl.Select(cond, a, b) — cond from `a > b` traced comparison. Should work (README shows Select with runtime condition).

Actually — is `txl.Select` correct name? README: "an FFI-backed expression constructor such as `txl.Select` accepts either wrapper directly" and "selected = txl.Select(choose_a, a_desc, b_desc)". Yes.

Alternatively use ptx max.f32: `txl.ptx.max.f32(dst, a, b)` needs DPS local. Select simpler.

Pass 1:

```python
        txl.assign(m, txl.float32(NEG_HUGE))
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                    if USE_SMEM:
                        txl.ptx.st.shared.f32(xs.ptr_to([col]), v)
                    txl.assign(m, txl.Select(v > m, v, m))
        mtot = cta_reduce(m, txl.float32(NEG_HUGE), fmax)
```

Wait: `it * TILE` — it is loop var, TILE python int. Fine. `j * BLOCK` python int, + tid traced. `col < n` — n is static python int; comparison traced vs constant, fine.

Hmm — `txl.assign(m, txl.Select(...))` where m local_scalar and condition v > m: fine.

Pass 2:

```python
        txl.assign(s, txl.float32(0.0))
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    if USE_SMEM:
                        txl.ptx.ld.shared.f32(v, xs.ptr_to([col]))
                    else:
                        txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                    # e = exp(v - mtot)
                    txl.ptx.ex2.approx.f32(e, (v - mtot) * txl.float32(LOG2E))
                    if USE_SMEM:
                        txl.ptx.st.shared.f32(xs.ptr_to([col]), e)
                    txl.assign(s, s + e)
        stot = cta_reduce(s, txl.float32(0.0), add)
```

Wait — in the smem path, pass 2 overwrites xs with e. Then pass 3 reads e from xs. But careful: pass 3 reads xs[col] where col assigned same as pass 2 — same thread wrote it. But hold on: is it the same thread? Pass 2 and pass 3 use identical col assignment loops → yes, same thread reads what it wrote. Own-write visibility without barrier: for shared memory, a thread's own writes are visible to itself (through L1/shared). Yes.

Actually hmm — but is it guaranteed that pass-2's store and pass-3's load to same address by same thread are ordered? Yes, program order within thread, same address, no race.

But wait, there's a hazard I missed in pass 2 itself: thread A reads xs[colA] while... each thread only reads locations it wrote in pass 1 and will overwrite with its own e. Thread B never reads colA. So no cross-thread hazard in pass 2. But — did all pass-1 stores complete before pass-2 reads? Pass-2 reads happen after cta_reduce's barriers. cta_reduce's first bar.sync is after warp_red (shfls only, no smem). Order: pass1 stores xs (guarded) → warp shfl → st red → bar.sync → ... Pass-2's ld.shared happens after bar #3 (the one after broadcast). All pass-1 xs stores precede the first bar, all pass-2 loads follow it. ✓ visible.

Pass 3:

```python
        inv = ...  # 1/stot
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    if USE_SMEM:
                        txl.ptx.ld.shared.f32(e, xs.ptr_to([col]))
                    else:
                        txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                        txl.ptx.ex2.approx.f32(e, (v - mtot) * txl.float32(LOG2E))
                    txl.ptx.st.global_.f32(y.ptr_to([r, col]), e * inv)
```

inv: `inv = txl.float32(1.0) / stot` — but that's a value used inside loop; compute once into local_scalar:

```python
        inv = txl.local_scalar("float32")
        txl.assign(inv, txl.float32(1.0) / stot)
```

Hmm, does `/` produce div.rn? Assume yes.

Also note: e is local_scalar reused; in non-smem pass3, compute exp into e. Fine.

Wait, one problem: in pass 2 smem path, I store e into xs AND accumulate. In pass 3 smem path, I load e. Good.

Also — dead concern: when NTILE could be 0? n>=1 → NTILE>=1. ✓.

Grid = rows: rows >= 1 presumably. If rows==0, grid 0 — not our concern.

Type detail: `col < n` — col is int32 traced, n python int constant → constant folded. `r` from cta_id — used in ptr_to([r, col]). The scale2 example did exactly this. ✓.

bar.sync first operand: example `txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(nthreads))`. I'll do `txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))`.

ld.shared.f32 DPS: `txl.ptx.ld.shared.f32(dst, ptr)` per example `txl.ptx[LD_S_F32](ss[0], sum_sq_smem.ptr_to([lane]))`. ✓.

st.shared.f32: `txl.ptx.st.shared.f32(ptr, val)` ✓.

ex2: `txl.ptx.ex2.approx.f32(e, arg)` — DPS with float dst. Should be OK.

One concern: `txl.local_scalar` vs `txl.alloc_local` — for scalars, local_scalar used in example with txl.assign. ✓.

Another: `txl.reinterpret(cur, ty="uint32")` — example used ty=txl.uint32 and ty="float32". Both forms shown. Use ty="uint32"/"float32".

Hmm, in the warp_sum example: `txl.reinterpret(acc[0], ty="uint32")` where acc is alloc_local array; acc[0] is element. For local_scalar m, use m directly.

Now — shflSync within `with txl.if_` guarded regions: our shfls are all outside guards except the second warp_red inside `if warp==0`, where all 32 lanes of warp 0 participate (the if condition is warp-uniform). ✓.

But there's a subtlety: the first shfl in the warp_red for `w` — inside `if warp==0`, and before it there's a divergent `if lane < NWARP` block that reconverges. Compiler handles via reconvergence. Fine (same as RMSNorm example).

One more check: `red.ptr_to([warp])` where warp = tid >> 5 — traced expr. ✓ (example used warp similarly).

Potential issue: `txl.Select` returns a value usable in assign and arithmetic. ✓.

Edge case n=1: NTILE=1; thread 0 loads x[0]; m = x[0]; other threads m=-FLT_MAX. reduce → max = x[0] (Select(x0 > -FLT_MAX)). Wait — careful with Select semantics when values equal: fine either way.

Sum: thread0: e=exp(0)=1; s=1; others 0. stot=1. inv=1. y=1. ✓.

Large negative row: all x = -100: m=-100, e=exp(0)=1 each, sum=n, y=1/n. ✓.

Mixed: x=[100, 0]: m=100, e=[1, e^-100≈3.7e-44 (denormal? e^-100 = 3.72e-44 — f32 min normal 1.18e-38, so 3.7e-44 is denormal!). With ex2.approx.f32 — does it flush denormals? ex2.approx.f32 supports denormal outputs? PTX: ex2.approx.f32 ... "denormal numbers: supported" I believe for ex2 approx it's implementation... Actually ex2.approx.ftz.f32 flushes; ex2.approx.f32 (non-ftz) preserves denormals on sm_20+. The result 3.7e-44 denormal, sum ≈ 1.0000..., y ≈ [1, 3.7e-44]. Correct. Even if flushed to 0, error is 1e-44 absolute — acceptable? "Numerically stable" — flushing to 0 gives y[1]=0 vs true 3.7e-44; relative error 1 on that element but it's negligible. Standard __expf does this. But to be safer on accuracy maybe use non-ftz ex2.approx (which I do). Fine.

Also compiler ftz: backend_config only sets arch; nvrtc default --ftz=false. Good.

What about `txl.float32(NEG_HUGE)` — -3.4028234663852886e38 representable in f32 (it's -FLT_MAX). ✓.

Now think about whether `txl.ptx.ex2.approx.f32` attribute access works — "ex2" then "approx" then "f32": `txl.ptx.ex2.approx.f32`. The txl.ptx object presumably builds instruction name from attribute path. Given `txl.ptx.shfl_sync.bfly.b32` works (three levels), `ex2.approx.f32` (three levels) should too.

Hmm, one risk: does txl.ptx validate instruction signatures? ld/st/mov/shfl/bar are used in examples. ex2 is a simple unary op. There's some risk the DPS wrapper requires certain handling for non-void results — but shfl_sync is non-void and handled with a destination local. I'll trust it.

Alternative safer exp: is there a documented `txl.exp`? Not documented in the README. txl.rsqrt is documented via example. Hmm. I'll go with ptx ex2 — it's plain PTX and sm_80-legal.

Also `txl.ptx.ld.global_.f32` with `_` due to Python keyword `global`. ✓ per scale2 example.

Now think about whether passing `combine` as Python closure works — yes, helpers are traced Python.

One thing: `txl.local_scalar("uint32")` peer — created inside cta_reduce, used in warp_red closure. Since warp_red defined inside cta_reduce, closure over peer. Called twice → the same peer local used in both; fine.

But careful: warp_red(acc) assigns to acc which may be `cur` or `w` locals. ✓.

Also the loop `for delta in (16,8,4,2,1)` is Python-time unrolled. ✓.

Check `txl.assign(cur, combine(acc, ...))` — wait in warp_red I wrote `txl.assign(acc, combine(acc, ...))`. ✓.

Now let me also double check bar.sync operand forms: `txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(nthreads))` — in my code: `txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))`. ✓.

Potential IR-contract issue: shfl inside `if warp == 0` — allowed? RMSNorm does the same. ✓.

Another potential issue: the trailing bar.sync after reading red[0] — all threads execute. ✓.

Also: the third bar.sync in cta_reduce — for the LAST cta_reduce call (sum), it's harmless extra sync.

Hmm, wait — actually there's a correctness hazard I should double check for pass 2 smem overwrite: pass 2 thread T reads xs[colT] and writes e to xs[colT]. But does thread T read xs[colT] before writing? Within the if block: ld then st. Program order, same thread, same address → read gets pass-1 value. ✓. No other thread touches colT. ✓.

But actually wait — is that true? Pass 1: thread T stores xs[colT]. Pass 2: thread T reads xs[colT]. Are these the same col assignment? Pass 1 loop: `col = base + j*BLOCK + tid` with serial(NTILE). Pass 2: identical structure. Yes same cols. ✓. And visibility of T's own pass-1 store to T's pass-2 load: same thread, no barrier needed for shared memory? In CUDA, a thread's own writes to shared memory are visible to its own subsequent reads without __syncthreads. Yes.

However — subtle: pass 2 reads happen after cta_reduce barriers anyway (mtot computed before pass 2 loop). So even cross-thread it'd be fine. ✓.

Pass 3 reads xs (now e values) written by same thread in pass 2, separated by cta_reduce barriers. ✓.

Now, non-smem path (n > 12288): pass 2 loads global, pass 3 loads global + recomputes exp. Correct.

USE_SMEM threshold: 48KB = 12288 floats. Static n. If n*4 <= 48*1024 → n <= 12288. But also smem total = red (16B) + xs. 12288*4 = 49152 = 48KB exactly, plus 16 bytes for red → 49168 > 49152! Default static+dynamic smem limit is 48KB total per block (without cudaFuncAttributeMaxDynamicSharedMemorySize). Hmm. Does smem_pool allocate dynamic smem and does txl set the attribute? Unknown. To be safe, use threshold 46KB: n <= 11000? Let me set USE_SMEM = n * 4 + 64 <= 48 * 1024, i.e. n <= 12240... let me just use n * 4 <= 47 * 1024 → n <= 12032. Simpler: USE_SMEM = n <= 12000. Eh, let me do `USE_SMEM = (n * 4 + NWARP * 4) <= 48 * 1024`.

Actually, 48KB is the limit for dynamic smem without opt-in; static smem limit is also 48KB combined. If txl emits dynamic smem and sets the attribute automatically (many DSLs do when needed), up to 163KB possible on A100. Can't rely. 48KB threshold safe. Could even stage in smem for n up to 163KB if attribute handled... skip, keep 48KB.

Alternatively, avoid the dual path complexity... but requirement says high performance; n often large (like 4096, 8192 — fits). Keep dual path.

Hmm, actually wait: should I also consider vectorized global loads (float4) in the non-smem path for large n? Large n means multiple CTAs... no, grid=rows, one CTA per row, 128 threads. For n = 100000, loop NTILE = 196 iterations of 4 unrolled scalar loads. Throughput: coalesced 128B per warp per load; with 4 in flight per thread, should saturate. It's fine.

Also consider: for very large n and few rows, occupancy is limited by grid — but that's the specified schedule (one CTA per row), fine.

Now, one more design check: `txl.serial(NTILE)` with NTILE Python int → static loop count. ✓ (example used (n+127)//128 python int... in scale2, n was python int static; yes `with txl.serial((n + 127) // 128) as it:`).

Indices in ptr_to for 2D tensor: x.ptr_to([r, col]) ✓.

Types: r is cta_id → int32. col int32. OK.

Now, `txl.uint32(FULL_MASK)` = 0xFFFFFFFF. FULL_MASK as Python int 0xFFFFFFFF — txl.uint32(0xFFFFFFFF) fine.

Let me also double-check `txl.Select` argument order: Select(cond, true_val, false_val). Standard. ✓.

Also arithmetic `(v - mtot) * txl.float32(LOG2E)` — traced operators on local_scalar values. The scale2 example: `v * txl.float32(2.0)` where v local_scalar. ✓. Subtraction similarly.

`s + e` where both local_scalars → traced add. ✓. `e * inv` ✓. `txl.float32(1.0) / stot` — division of constant by traced value. RMSNorm: `ss[0] / txl.float32(hidden_size)` — traced / constant. Constant / traced should also work; to be safer, `txl.assign(inv, txl.float32(1.0)); txl.assign(inv, inv / stot)` — same thing. Or compute `e / stot` per element — one division per element, slower but negligible vs memory? Div is ~20 cycles throughput-heavy... per element it's fine actually but let's precompute inv once.

Hmm, `txl.float32(1.0) / stot` — I'll write it; risk minimal.

Now potential concern about `m` init: `txl.assign(m, txl.float32(NEG_HUGE))`. NEG_HUGE = -3.402823466e38. Fine.

Now about "check_ir=True" default — our kernel should pass the low-level IR contract. Things like shfl needing convergence, barriers executed uniformly — we comply.

One more: the extra trailing bar.sync inside cta_reduce after `ld.shared res` — hmm, is it OK to have bar.sync where all threads reach it? Yes.

But wait, actually do I even need it? Next use of red is in the NEXT cta_reduce's `st red[warp]` by lane 0 of each warp. Between: current cta_reduce's ld red[0] (all threads), then pass-2 loop (long), then next cta_reduce's warp shfl, then st red[warp]. Could a fast thread's st red[warp] race a slow thread's ld red[0]? The ld red[0] happens right before... threads could be delayed. In practice the pass-2 loop work separates them but no guarantee. Keep the extra bar. Cost trivial.

Actually, careful ordering issue: bar.sync after the ld — I have `ld.shared res; bar.sync; return res`. Fine.

Hmm — wait, actually there's a subtle issue: is `res` a fresh local per cta_reduce call, returned as a value (mtot, stot). `txl.ptx.ld.shared.f32(res, ...)` then return res — the returned expression is the local's value. ✓.

Let me now also think: do we need `txl.alloc_local` anywhere? No arrays needed.

Also: kernel name/args must match interface: `def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32"))`. ✓.

Potential issue: annotations with Python ints rows, n — static shapes. The example `scale2` did exactly this. ✓.

Also `txl.device_entry(...)` flat form with backend_config kwarg — per the required interface. ✓.

Let me now also handle possible concern: `with txl.if_(col < n), txl.then_():` — combining in one with? The scale2 example: `with txl.if_(col < n), txl.then_():`. ✓ same form.

Now write out full code carefully.

Also: `warp = tid >> 5` — traced int shift. RMSNorm example used `tid >> 5`. ✓. `lane = tid & 31` ✓.

One more perf thought: use `ld.global.nc` (non-coherent / __ldg)? x is read-only. ld.global.nc.f32 could improve L1 caching. Spell: txl.ptx.ld.global.nc.f32 — 4 levels. Risky? It's plain PTX. I'll keep plain ld.global (which on modern archs caches in L1 anyway for read-only... actually ld.global caches in L1 only with .nc or on Ampere default? On Volta+, generic ld.global is cached in L1). Keep plain.

Final code:

```python
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
NEG_HUGE = -3.4028234663852886e38
LOG2E = 1.4426950408889634


def make_kernel(rows: int, n: int):
    BLOCK = 128
    NWARP = BLOCK // 32
    UNROLL = 4
    TILE = BLOCK * UNROLL
    NTILE = (n + TILE - 1) // TILE
    USE_SMEM = (n * 4 + NWARP * 4) <= (48 * 1024)

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
            backend_config={"cuda": {"arch": "sm_80"}},
        )

        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        red = smem.alloc([NWARP], "float32")
        if USE_SMEM:
            xs = smem.alloc([n], "float32")

        v = txl.local_scalar("float32")
        e = txl.local_scalar("float32")
        m = txl.local_scalar("float32")
        s = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")

        def cta_reduce(val, ident, combine):
            cur = txl.local_scalar("float32")
            peer = txl.local_scalar("uint32")
            txl.assign(cur, val)

            def warp_red(acc):
                for delta in (16, 8, 4, 2, 1):
                    txl.ptx.shfl_sync.bfly.b32(
                        peer,
                        txl.reinterpret(acc, ty="uint32"),
                        txl.uint32(delta),
                        txl.uint32(31),
                        txl.uint32(FULL_MASK),
                    )
                    txl.assign(acc, combine(acc, txl.reinterpret(peer, ty="float32")))

            warp_red(cur)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            with txl.if_(warp == 0), txl.then_():
                w = txl.local_scalar("float32")
                with txl.if_(lane < NWARP), txl.then_():
                    txl.ptx.ld.shared.f32(w, red.ptr_to([lane]))
                with txl.else_():
                    txl.assign(w, ident)
                warp_red(w)
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx.st.shared.f32(red.ptr_to([0]), w)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            out = txl.local_scalar("float32")
            txl.ptx.ld.shared.f32(out, red.ptr_to([0]))
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            return out

        def fmax(a, b):
            return txl.Select(a > b, a, b)

        def fadd(a, b):
            return a + b

        # pass 1
        txl.assign(m, txl.float32(NEG_HUGE))
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                    if USE_SMEM:
                        txl.ptx.st.shared.f32(xs.ptr_to([col]), v)
                    txl.assign(m, txl.Select(v > m, v, m))

        mtot = cta_reduce(m, txl.float32(NEG_HUGE), fmax)

        # pass 2
        txl.assign(s, txl.float32(0.0))
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    if USE_SMEM:
                        txl.ptx.ld.shared.f32(v, xs.ptr_to([col]))
                    else:
                        txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                    txl.ptx.ex2.approx.f32(e, (v - mtot) * txl.float32(LOG2E))
                    if USE_SMEM:
                        txl.ptx.st.shared.f32(xs.ptr_to([col]), e)
                    txl.assign(s, s + e)

        stot = cta_reduce(s, txl.float32(0.0), fadd)
        txl.assign(inv, txl.float32(1.0) / stot)

        # pass 3
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    if USE_SMEM:
                        txl.ptx.ld.shared.f32(e, xs.ptr_to([col]))
                    else:
                        txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                        txl.ptx.ex2.approx.f32(e, (v - mtot) * txl.float32(LOG2E))
                    txl.ptx.st.global_.f32(y.ptr_to([r, col]), e * inv)

    return softmax_txl
```

Wait — one issue in pass 2/3 smem path: `xs.ptr_to([col])` for ld.shared and st.shared — st then ld same address in pass2; but pass 2 stores e to xs BEFORE pass 3 loads. But hold on — pass 3 loads xs[col], which now contains e. But in pass 2, I store e and then pass 3 loads it — fine. However, in pass 2 non-smem path nothing stored.

Another wait — pass 2 smem: I read v from xs then write e to xs same location. But OTHER threads in pass 2 — do any read xs[col] written by another thread? No, each thread only accesses its own cols in every pass. ✓.

Hmm — but actually one subtle thing: pass 3 in smem path loads e from xs — thread's own write from pass 2, separated by cta_reduce barriers. Definitely visible. ✓.

Now, `it * TILE`: `it` is loop var; TILE python int 512. `base = it * TILE` — traced int32 mult. `j * BLOCK` python. `col = base + j*BLOCK + tid` ✓.

Issue: `txl.assign(inv, txl.float32(1.0) / stot)` — if stot could be 0? Only if n==0 or all exp underflow — exp(x-max) includes at least one exp(0)=1, so stot >= 1. ✓.

Now, about `txl.ptx.ex2.approx.f32(e, arg)` — DPS with e local_scalar float32. I'm reasonably confident. Alternatively could there be an issue where ptx unary ops need different call form? shfl_sync took (dst, src...). ld took (dst, ptr). ex2: (dst, src). Consistent.

Alternative to reduce risk: compute exp via txl.exp? Not documented. Keep ex2.

Hmm, actually, one more consideration about accuracy of ex2.approx.f32 combined with multiply by LOG2E: standard __expf. Max ulp error ~2. Fine.

Should I worry about the IR checker complaining about `shfl_sync` inside `if warp == 0` (convergence)? RMSNorm does exactly this. ✓.

What about `txl.ptx.bar.sync` count operand — must be multiple of warp size, 128 ✓.

One thing to double check: `red = smem.alloc([NWARP], "float32")` — NWARP=4 python int. ✓ (example: smem.alloc([hidden_size], "float32")).

Another check: in cta_reduce, `txl.assign(w, ident)` where ident is a traced constant like txl.float32(NEG_HUGE). ✓.

Also `combine(acc, ...)` where acc is local_scalar — reading it yields value. ✓.

Edge: `txl.Select(a > b, a, b)` with a,b local_scalars → comparison traced. In pass 1, `txl.Select(v > m, v, m)`. ✓.

Potential type issue: `tid & 31` — int traced & python int. Example did it. ✓.

Now — should the `if USE_SMEM:` at trace time with `xs` defined conditionally cause issues? It's plain Python; when False, xs undefined but never referenced (all references inside USE_SMEM branches). ✓.

One more possible concern: for n small (< TILE), NTILE = 1 and guards handle. For n < BLOCK e.g. n=1: pass1 thread0 only. ✓.

Performance thought: with unroll 4 and guarded loads, for full tiles the guards are `col < n` with col up to base+384+127 < n — compiler can't hoist since n runtime? n is STATIC python int! So `col < n` is `base + j*128 + tid < n` — traced comparison, loop bounds static. ptxas may not optimize much, but branch is cheap and warp-uniform-ish (per-iteration, tail only). Fine.

Actually, since n is static, I could specialize the loop to avoid guards for full tiles and only guard the last tile. Micro-opt; adds code. The guard cost is small (predicated loads). Skip.

One more: should I use `ld.global.nc.f32` for x loads in non-smem path... keep plain.

Also consider using `.v4` vector loads when n % (4*BLOCK) == 0 or n % 4 == 0 for extra perf. I'll skip for robustness — the unrolled scalar version achieves good bandwidth.

Hmm, wait — actually let me reconsider one potential correctness trap: shfl_sync.bfly with clamp operand 31. In the RMSNorm example, for warp_sum they used 31 as clamp. For bfly mode, clamp is ignored but required operand. ✓.

Another trap: `txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)` — ptr_to on smem tensor with traced index. Example: `txl.ptx[ST_S_F32](sum_sq_smem.ptr_to([warp]), ss[0])`. ✓.

I think the code is solid. One more look at `make_kernel` signature and return. ✓.

Also the interface says "one kernel is traced per (rows, n)" — our factory computes NTILE etc. at factory time. ✓.

Let me reconsider the `USE_SMEM` threshold: red is 4 floats = 16 bytes; xs = n*4. Total = 4n+16 ≤ 49152 → n ≤ 12284. My condition `n * 4 + NWARP * 4 <= 48 * 1024` → n ≤ 12284. OK.

But does smem_pool allocation count toward the 48KB static limit or dynamic? If txl uses dynamic shared memory with opt-in attribute automatically, great; if static, 48KB total applies. Either way ≤48KB is safe. ✓.

Also note: for huge n, non-smem path does 3 global reads of x total (pass1, pass2, pass3) — wait pass3 recomputes exp from global. Yes 3 reads 1 write. L2 on A100 is 40MB; rows*n*4 may exceed but streams fine.

Could improve non-smem path: pass 2 could store e to y and pass 3 scale y in place (read+write y instead of read x + exp). Similar cost. Keep as is.

Double-check `txl.serial(NTILE)` — if NTILE is a Python int ≥ 1. ✓.

One concern: variable shadowing `it` and `base`, `col` across three serial loops — Python rebinding, fine in tracing.

Also: `e * inv` — traced mult of two local_scalars ✓.

Final check on `txl.float32(1.0) / stot`: stot is a local_scalar returned from cta_reduce. Division: TIR fdiv. Should lower to div.rn.f32 (or div.approx if fast-math — we don't set fast math). ✓.

Also, hmm — `txl.assign(m, txl.float32(NEG_HUGE))` then in loop `txl.Select(v > m, v, m)` — reads m. ✓.

I'm also wondering whether `txl.ptx.ex2.approx.f32` might collide with reserved words... "ex2", "approx", "f32" none are Python keywords. ✓. And `txl.ptx.ld.shared.f32` — "shared" not keyword ✓ (example used string constant, but attribute should work; e.g. `txl.ptx.st.global_.f32` used attribute). Actually for safety, use the getitem string form for all instructions not shown in attribute form? Attribute form shown: st.global_.f32, ld.global_.f32 (scale2), mov.b32, shfl_sync.bfly.b32, bar.sync. ld.shared.f32 / st.shared.f32 via attribute: `txl.ptx.ld.shared.f32` — "ld" ok, "shared" ok, "f32" ok. I'll use attribute form; it's the same mechanism.

Hmm, one risk with ex2.approx.f32: some DSL ptx wrappers require exact PTX opcode validation against a known list; ex2 is standard PTX so likely included. Accept.

Alternative fallback thought: implement exp via `txl.exp`? If txl.ptx fails, everything fails. I must choose one. The instructions say "Plain f32 scalar ops" fine — ex2.approx is a plain f32 scalar op. Go.

One more consideration — the problem says "one CTA (128 threads) per row" and grid=rows, block=128 in the given entry. ✓.

Also should I add `kernel_attrs` like min_blocks_per_sm? Not needed.

Write final answer.

## content
```python
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
NEG_HUGE = -3.4028234663852886e38  # -FLT_MAX: identity for max reduction
LOG2E = 1.4426950408889634


def make_kernel(rows: int, n: int):
    BLOCK = 128
    NWARP = BLOCK // 32
    UNROLL = 4
    TILE = BLOCK * UNROLL
    NTILE = (n + TILE - 1) // TILE
    # Stage the row in shared memory (and exp() results) when it fits in the
    # default 48KB shared-memory budget; otherwise re-read from global.
    USE_SMEM = (n * 4 + NWARP * 4) <= (48 * 1024)

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
            backend_config={"cuda": {"arch": "sm_80"}},
        )

        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        red = smem.alloc([NWARP], "float32")
        if USE_SMEM:
            xs = smem.alloc([n], "float32")

        v = txl.local_scalar("float32")
        e = txl.local_scalar("float32")
        m = txl.local_scalar("float32")
        s = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")

        def cta_reduce(val, ident, combine):
            """Block-wide reduction: warp butterfly, warp 0 across warps, broadcast."""
            cur = txl.local_scalar("float32")
            peer = txl.local_scalar("uint32")
            txl.assign(cur, val)

            def warp_red(acc):
                for delta in (16, 8, 4, 2, 1):
                    txl.ptx.shfl_sync.bfly.b32(
                        peer,
                        txl.reinterpret(acc, ty="uint32"),
                        txl.uint32(delta),
                        txl.uint32(31),
                        txl.uint32(FULL_MASK),
                    )
                    txl.assign(acc, combine(acc, txl.reinterpret(peer, ty="float32")))

            warp_red(cur)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            with txl.if_(warp == 0), txl.then_():
                w = txl.local_scalar("float32")
                with txl.if_(lane < NWARP), txl.then_():
                    txl.ptx.ld.shared.f32(w, red.ptr_to([lane]))
                with txl.else_():
                    txl.assign(w, ident)
                warp_red(w)
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx.st.shared.f32(red.ptr_to([0]), w)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            out = txl.local_scalar("float32")
            txl.ptx.ld.shared.f32(out, red.ptr_to([0]))
            # Protect reuse of `red` by the next reduction.
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            return out

        def fmax(a, b):
            return txl.Select(a > b, a, b)

        def fadd(a, b):
            return a + b

        # ---- pass 1: row max (stage the row in smem when it fits) ----
        txl.assign(m, txl.float32(NEG_HUGE))
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                    if USE_SMEM:
                        txl.ptx.st.shared.f32(xs.ptr_to([col]), v)
                    txl.assign(m, txl.Select(v > m, v, m))

        mtot = cta_reduce(m, txl.float32(NEG_HUGE), fmax)

        # ---- pass 2: sum of exp(x - max); keep exp values for pass 3 ----
        txl.assign(s, txl.float32(0.0))
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    if USE_SMEM:
                        txl.ptx.ld.shared.f32(v, xs.ptr_to([col]))
                    else:
                        txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                    # exp(v - mtot) via ex2: stable, args are <= 0.
                    txl.ptx.ex2.approx.f32(e, (v - mtot) * txl.float32(LOG2E))
                    if USE_SMEM:
                        # Each thread only rewrites slots it owns; safe in place.
                        txl.ptx.st.shared.f32(xs.ptr_to([col]), e)
                    txl.assign(s, s + e)

        stot = cta_reduce(s, txl.float32(0.0), fadd)
        txl.assign(inv, txl.float32(1.0) / stot)

        # ---- pass 3: normalize and store ----
        with txl.serial(NTILE) as it:
            base = it * TILE
            for j in range(UNROLL):
                col = base + j * BLOCK + tid
                with txl.if_(col < n), txl.then_():
                    if USE_SMEM:
                        txl.ptx.ld.shared.f32(e, xs.ptr_to([col]))
                    else:
                        txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                        txl.ptx.ex2.approx.f32(e, (v - mtot) * txl.float32(LOG2E))
                    txl.ptx.st.global_.f32(y.ptr_to([r, col]), e * inv)

    return softmax_txl
```
