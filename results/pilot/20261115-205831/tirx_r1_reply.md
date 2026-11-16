# tirx round 1

model=glm-5.3 in=7736 out=13353 cost=0.06695 latency=57.2s finish=stop effort=high usage={"completion_tokens": 13353, "completion_tokens_details": null, "prompt_tokens": 7736, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 7680, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 11304, "total_tokens": 21089}

## reasoning
Need to write a row-wise softmax kernel, sm_80, one CTA per row, 128 threads, static shapes (rows, n known at trace time). Correct for any n>=1. Numerically stable.

Approach: each CTA handles one row. Read row into shared memory (float32), compute max via warp shuffle reduction + cross-warp via smem, then compute exp sums, then normalize and store.

Plan:
- smem pool: x_smem alloc [n] float32, and red_smem [4] float32 (for 4 warps).
- Pass 1: loop over n in strides of 128, load x global, store to smem, track thread-local max.
- Reduce max: warp butterfly shuffle on max (use shfl.sync.bfly.b32 with reinterpret), lane 0 writes to red_smem[warp], bar.sync, then warp 0 (or all threads) loads 4 values and reduces. Simpler: all threads read 4 values from smem and compute local max, then done — but that requires all threads read after a barrier; that's fine: after bar.sync, each thread loads red_smem[0..3] and computes max. Actually only need thread-local max for all threads since exp needs max in each thread. Have each thread compute its own max over the 4 warp maxima. That works and avoids another reduction.

Alternative simpler: after bar.sync, every thread loads 4 smem values, computes max — broadcast reads, fine.

- Pass 2: loop over row again (from smem), compute e = exp(x - max), accumulate sum, store e back to smem (overwrite x_smem).
- Reduce sum same way.
- Pass 3: read e from smem, divide by sum, store to y global.

Vectorization: could use vectorized loads if n % 4 == 0. To keep it simple but performant, maybe do a general scalar loop; or branch: if n % 4 == 0 use v4 loads. For simplicity and correctness, I'll write scalar loads/stores with ld.global.f32 — but performance "aim for high performance". Scalar loads of f32 give 4B per thread per transaction; with 128 threads that's 512B per warp-instruction... Actually each warp loading 32 consecutive floats = 128B = one 128B transaction, fine coalescing. Vectorized would reduce instruction count but scalar is already coalesced. Acceptable.

But maybe implement vectorization when n % 4 == 0 at trace time (Python-level if since n is static). That's doable: make_kernel receives n as int, so we can specialize Python-side. Let me do that: vec = 4 if n % 4 == 0 else 1. Use ld.global.v4.f32 with uint32 words? For float32, ld.global.v4.f32 exists ("ld.global.v4.f32"). Yes, PTX supports ld.global.v4.f32. Store st.global.v4.f32 too. And ld.shared.v4.f32/st.shared.v4.f32 as in RMSNorm example.

Let me write:

```python
import math
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
NTHREADS = 128
NWARP = 4

def make_kernel(rows: int, n: int):
    vec = 4 if n % 4 == 0 else 1
    nvec = (n + vec - 1) // vec  # number of vec-elements
    iters = (nvec + NTHREADS - 1) // NTHREADS

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        x_smem = smem.alloc([n], "float32")
        warp_red = smem.alloc([NWARP], "float32")

        m = txl.local_scalar("float32")  # local max
        s = txl.local_scalar("float32")  # local sum
        row_max = txl.local_scalar("float32")
        row_sum = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")
        v = txl.alloc_local([vec], "float32")

        # helpers
        def warp_reduce_max(acc):
            peer = txl.local_scalar("uint32")
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer, txl.reinterpret(acc, ty="uint32"),
                    txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
                p = txl.reinterpret(peer, ty="float32")
                txl.assign(acc, txl.max(acc, p))
```

Wait — is there txl.max? Not documented. Use Select: txl.Select(acc > p, acc, p). Or use PTX max.f32: txl.ptx.max.f32(dst, a, b). PTX has max.f32 instruction. DPS form: txl.ptx.max.f32(acc, acc, p). Hmm — destination same as source, should be fine in DPS. Actually safer to use a temp: txl.ptx.max.f32(acc, acc, peer_f). DPS form with dst==src should work.

Similarly min not needed. For sum: acc = acc + p.

But careful: local_scalar defined inside helper each call — creating a new local per call is fine since helpers are traced once each call site.

Actually in RMSNorm example, warp_sum takes acc as an array. I'll write two helpers warp_max and warp_sum operating on local_scalar.

Cross-warp: lane 0 stores to warp_red[warp]; bar.sync(0, 128); then each thread reads warp_red[0..NWARP-1] and reduces locally.

For max: initialize m to -inf: use txl.float32("-inf")? Better to just load first value... use -3.402823466e38 (float min). txl.float32 accepts a float; can pass float("-inf")? IR float constant inf might be supported. Safer: initialize with the first loaded value approach, or use a very negative finite constant -3.402823e38. Since x ~ 100, fine. Use txl.float32(-3.4028234663852886e38).

Pass 1 with vec:

```python
        txl.assign(m, txl.float32(-3.4028234663852886e38))
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec  # element index
            with txl.if_(idx < n), txl.then_():
                if vec == 4:
                    txl.ptx.ld.global.v4.f32(v[0], v[1], v[2], v[3], x.ptr_to([r, idx]))
                else:
                    txl.ptx.ld.global.f32(v[0], x.ptr_to([r, idx]))
                for k in range(vec):
                    txl.ptx.st.shared.f32(x_smem.ptr_to([idx + k]), v[k])
                    txl.ptx.max.f32(m, m, v[k])
```

Hmm — does txl.ptx have ld.global.v4.f32 spelled that way? RMSNorm uses LD_S_V4 = "ld.shared.v4.f32" via txl.ptx[LD_S_V4](...). And ld.global.v4.b32 via txl.ptx[LD_G_V4]. So txl.ptx["ld.global.v4.f32"] with 4 dests, 1 src pointer. Good. Also direct attribute access likely works: txl.ptx.ld.global.v4.f32 — but "global_" vs "global"? Example uses ld.global_.f32 (with underscore because "global" is a Python keyword). For v4: txl.ptx["ld.global.v4.f32"] via indexing is safest.

Also st.shared.f32 vs st.shared_.f32? "shared" is not a keyword, example uses txl.ptx[ST_S_F32] with ST_S_F32 = "st.shared.f32". OK, use index style for everything.

Indices: idx = it * NTHREADS * vec + tid * vec. Since it and tid are int32 exprs; ptr_to index must be int? n and rows are static ints in the Tensor shape, index expressions int32 fine.

Note x_smem.ptr_to — smem alloc returns tensor; ptr_to([idx]) gives shared pointer. Good, matches example.

Pass 2 needs barrier after max reduction:

```python
        # warp max reduce
        warp_max(m)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx["st.shared.f32"](warp_red.ptr_to([warp]), m)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))
        txl.assign(row_max, txl.float32(-3.4...e38))
        for w in range(NWARP):
            t = txl.local_scalar("float32")
            txl.ptx["ld.shared.f32"](t, warp_red.ptr_to([w]))
            txl.ptx.max.f32(row_max, row_max, t)
```

Better: all threads compute row_max from smem broadcast reads. Fine.

Pass 2: exp sum. exp: is there txl.exp or txl.idioms? txl.rsqrt exists per example. Likely txl.exp exists too? Not documented. Use PTX ex2.approx.f32: exp(x) = ex2(x * log2(e)). txl.ptx.ex2.approx.f32(dst, src) — DPS? ex2.approx has 2 operands (dst, src). Hmm, does txl.ptx support non-commutative 2-operand in DPS? DPS form: destination-first, so txl.ptx.ex2.approx.f32(dst, src). But I'm not sure txl.ptx auto-converts: "PTX and CUDA instructions are spelled through txl.ptx". The RMSNorm example uses mov.b32(dst, src) which is 2-operand. So ex2.approx.f32(dst, src) should work.

Precision: ex2.approx has ~2 ulp error; acceptable for softmax typically. But requirement says "numerically stable for large |x| (x ~ 100)". Stability refers to subtracting max — fine. Use ex2.approx.f32 with multiply by log2e = 1.4426950408889634. Or use lg2... For accuracy maybe use txl.exp if exists. Risky to invent API. The guide says "Only import tirx_kernels.tirx_lite (and math if needed)". math is only host-side. I'll use ex2.approx.f32 — standard for fast softmax. Alternatively there might be txl.idioms... unknown. Use PTX.

Actually, ex2.approx.f32 accuracy: max error 2 ulp, fine.

Also handle ftz? Default fine.

Pass 2:

```python
        txl.assign(s, txl.float32(0.0))
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec
            with txl.if_(idx < n), txl.then_():
                for k in range(vec):
                    e = txl.local_scalar("float32")
                    txl.ptx["ld.shared.f32"](e, x_smem.ptr_to([idx + k]))
                    txl.ptx.sub.f32(e, e, row_max)
                    txl.ptx.mul.f32(e, e, txl.float32(LOG2E))
                    txl.ptx.ex2.approx.f32(e, e)
                    txl.assign(s, s + e)
                    txl.ptx["st.shared.f32"](x_smem.ptr_to([idx + k]), e)
```

Hmm — DPS with dst==src for sub/mul: mov.b32 in example had distinct dst/src but dst==src aliasing in PTX is legal (same register). Should be OK. Alternatively use arithmetic expressions: e = e - row_max creates new SSA. But e is a local_scalar; txl.assign(e, e - row_max) works with expression operators (RMSNorm uses ss[0] + xf[kv]*xf[kv]). Simpler:

```python
txl.assign(e, e - row_max)
t2 = e * txl.float32(LOG2E)
txl.ptx.ex2.approx.f32(e, t2)
```

Then sum reduction for s, same as max but with add.

Pass 3: read e from smem, divide by row_sum, store to y.

Division: use reciprocal? txl.ptx.rcp.approx.f32 or exact div via expression e / row_sum. For accuracy use rcp.rn? PTX has div.rn.f32 (default IEEE). Expression division should compile to div.rn.f32 or fast depending. Use mul by reciprocal computed once: inv = 1.0/row_sum via rcp? Precision: rcp.approx has ~1 ulp... Actually for softmax, dividing each by sum: use exact div expression `e / row_sum` — compiler may emit div.rn.f32 (slow) or with fast math div.approx. I'll compute inv = txl expression: txl.assign(inv, txl.float32(1.0) / row_sum) then multiply. Fine.

Guards: threads with idx >= n still participate in shuffles and bar.sync — must ensure bar.sync is outside if-guards and all threads reach it. Serial loops with iters computed such that all threads execute the same number of iterations — yes, iters is uniform. The `if idx < n` guards are inside. Warp shuffles executed uniformly. Good.

Edge case n < 128: iters = 1, guards fine. vec=1 path since n%4 may be 0 even for n<128 (e.g., n=8, vec=4, iters=1). Fine.

Edge case: n=1: vec=1, iters=1, only tid 0 loads. row_max = x. s = 1. y = 1. Good.

One concern: iters when nvec... for vec=4, nvec = n//4 exactly (n%4==0), iters = ceil(nvec/128).

Also `x.ptr_to([r, idx])` — index types: r int32, idx int32. Rows tensor shape (rows, n) with static ints. Fine.

Vector load address alignment: ld.global.v4.f32 requires 16B alignment. x row base: row-major, n%4==0 means row start offset r*n*4 bytes — multiple of 16. idx multiple of 4. Good. Same for smem v4 if used. For smem I use scalar st.shared.f32 per element — could vectorize smem too but scalar is fine; actually for performance let's vectorize smem too when vec==4: ld.shared.v4.f32 / st.shared.v4.f32. Let me do that to reduce instruction count.

Pass 1 vec=4:
```python
txl.ptx["ld.global.v4.f32"](v[0], v[1], v[2], v[3], x.ptr_to([r, idx]))
txl.ptx["st.shared.v4.f32"](x_smem.ptr_to([idx]), v[0], v[1], v[2], v[3])
for k in range(4): txl.ptx.max.f32(m, m, v[k])
```
Shared alignment: x_smem is [n] float32, idx multiple of 4 → 16B aligned within pool; pool base aligned presumably 16B. warp_red allocated after — allocation alignment should be at least 4; the v4 stores happen at offsets multiple of 16 from base. Should be fine.

Pass 2 vec=4: need e values for 4 elements, then st.shared.v4.f32. Compute into local array ev[4]:

```python
ev = txl.alloc_local([4], "float32")
for k in range(4):
    ld.shared.f32 ev[k] <- x_smem[idx+k]  # or v4 load
    assign ev[k] = ev[k] - row_max
    ex2...
```
Simpler: v4 load into v[0..3], then compute ev[k] = ex2(v[k]*log2e - row_max*log2e)... note (x - max)*log2e. Then sum. Then st.shared.v4.f32.

```python
txl.ptx["ld.shared.v4.f32"](v[0], v[1], v[2], v[3], x_smem.ptr_to([idx]))
for k in range(4):
    e = (v[k] - row_max) * LOG2E
    txl.ptx.ex2.approx.f32(ev[k], e)
    txl.assign(s, s + ev[k])
txl.ptx["st.shared.v4.f32"](x_smem.ptr_to([idx]), ev[0], ev[1], ev[2], ev[3])
```

But wait — ex2.approx.f32 DPS: does txl.ptx support instructions with exactly (dst, src)? mov.b32(dst, src) yes. Should be fine.

Pass 3 vec=4:
```python
txl.ptx["ld.shared.v4.f32"](v[0..3], x_smem.ptr_to([idx]))
for k: o[k] = v[k] * inv
txl.ptx["st.global.v4.f32"](y.ptr_to([r, idx]), o[0..3])
```

Now the guard `idx < n`: for vec=4 and n%4==0, idx<n suffices for full vector validity. For vec=1 same.

Hmm, one issue: `with txl.if_(idx < n)` where idx is an expression — fine.

Also shared memory size: n floats. For very large n (e.g., 100k), smem exceeds 48KB and kernel fails. Should I handle large n without smem staging (re-read from global)? The task says "n is often not a multiple of 128" but no bound. Typical softmax kernels cap smem. To be robust for any n, I could fall back to a no-smem variant when n*4 > 48KB (or 164KB with opt-in, but keep it simple: 48KB → n <= 12288). Python-side choose variant since n is static. That adds code but improves correctness for ANY n. Requirement: "Correct for ANY n >= 1". A row must fit in smem otherwise the kernel breaks. Let me implement two variants:

Variant A (n <= 12288): smem staging (3 passes but only 1 global read + 1 global write).
Variant B (large n): no smem staging, two global read passes (max pass, then exp/sum pass, then third pass normalize writing). Reads global 3 times total (pass1 read, pass2 read, pass3 read... pass3 can re-read global and recompute exp, or store exp... no smem for exp). Pass B: pass1 compute max (global reads), pass2 compute sum of exp (global reads), pass3 compute exp/denom and write (global reads). 3 reads + 1 write.

That's a fair amount of duplicated code. I could unify: write one body parameterized by use_smem flag with Python-level if. Let me structure:

```python
def make_kernel(rows, n):
    vec = 4 if n % 4 == 0 else 1
    nvec = n // vec if vec == 4 else n
    iters = (nvec + 127) // 128
    USE_SMEM = n * 4 <= 48 * 1024
    ...
    @txl.kernel()
    def softmax_txl(x, y):
        ...
        smem = txl.smem_pool()
        if USE_SMEM:
            x_smem = smem.alloc([n], "float32")
        warp_red = smem.alloc([4], "float32")
```

Hmm, careful: `x_smem` defined conditionally in Python — fine, tracing is Python.

Pass 1:
```python
with txl.serial(iters) as it:
    idx = (it * NTHREADS + tid) * vec
    with txl.if_(idx < n), txl.then_():
        load global into v[0..vec-1]
        if USE_SMEM: store to smem
        for k: m = max(m, v[k])
```

Pass 2:
```python
with txl.serial(iters) as it:
    idx = ...
    with txl.if_(idx < n), txl.then_():
        if USE_SMEM:
            load from smem into v
        else:
            load from global into v
        for k: ev[k] = ex2((v[k]-row_max)*log2e); s += ev[k]
        if USE_SMEM:
            store ev to smem
```

Pass 3:
```python
with txl.serial(iters) as it:
    idx = ...
    with txl.if_(idx < n), txl.then_():
        if USE_SMEM:
            load ev from smem
        else:
            load from global into v, recompute ev[k] = ex2(...)
        for k: o[k] = ev[k] * inv
        store global
```

Actually for the non-smem variant pass 3, could fuse pass2/pass3? No, need the sum first.

Note when !USE_SMEM, pass 2 loads from global, computes ev but doesn't store — dead? No, s accumulates. ev values discarded (well, computed but unused except sum). Pass 3 recomputes. Fine.

To reduce code, define helper load_row(dst, idx) that loads from smem or global. And store helpers. Let me write it cleanly.

Let me also double check txl.ptx.max.f32 spelling: PTX instruction "max.f32". txl.ptx.max might not exist as attribute chain if... mov.b32 works as attribute; max.f32 → txl.ptx.max.f32 should work ("max" isn't reserved). But to be safe use txl.ptx["max.f32"](...). The example used both styles: txl.ptx.st.global_.f32 attribute and txl.ptx[LD_G_V4] index. Index style with full string is safest.

shfl: txl.ptx.shfl_sync.bfly.b32 used in example. Keep that.

bar.sync: txl.ptx.bar.sync(txl.uint32(0), txl.uint32(128)). Keep.

ex2.approx.f32: txl.ptx["ex2.approx.f32"](dst, src).

Hmm, is there an issue that txl.ptx[...] indexing works for any instruction? The example used it, yes.

Now the warp max helper:

```python
def warp_max(acc):
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer, txl.reinterpret(acc, ty="uint32"),
            txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.ptx["max.f32"](acc, acc, txl.reinterpret(peer, ty="float32"))
```

Wait, txl.reinterpret(value, ty=txl.uint32) — example: txl.reinterpret(acc[0], ty="uint32") and txl.reinterpret(peer, ty="float32"). Both forms shown. Use ty="uint32" string form as in the example body.

Note: local_scalar helper functions defined inside traced function: creating txl.local_scalar inside a helper called multiple times — each call creates a new local var, fine. But careful: helpers must be defined inside kernel function or can be module-level? They call txl.* which requires active device region/IRBuilder — module-level helpers traced during kernel trace... The guide says "Emission and source locations also work for helpers". RMSNorm defines helpers inside the kernel function. I'll define helpers inside the kernel body (nested), like RMSNorm. Actually warp_max and warp_sum are called at fixed points, so define them inside.

Wait, one subtlety: helpers defined inside the kernel body are Python functions; when called they emit code. Fine.

Alternatively define one generic reduction with an op parameter... just write two.

warp_sum:
```python
def warp_sum(acc):
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(acc, ty="uint32"),
            txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.assign(acc, acc + txl.reinterpret(peer, ty="float32"))
```

Cross-warp broadcast:
```python
def cta_reduce(op):  # returns final value in each thread's scalar
```
Hmm, different ops. Just inline both:

```python
warp_max(m)
with txl.if_(lane == 0), txl.then_():
    txl.ptx["st.shared.f32"](warp_red.ptr_to([warp]), m)
txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))
txl.assign(row_max, txl.float32(NEG_BIG))
for w in range(NWARP):
    t = txl.local_scalar("float32")
    txl.ptx["ld.shared.f32"](t, warp_red.ptr_to([w]))
    txl.ptx["max.f32"](row_max, row_max, t)
```

Creating local scalar inside a Python for loop over range(NWARP) — creates 4 locals, fine. Or reuse one t declared before loop. Do that.

Similarly for sum:
```python
warp_sum(s)
with txl.if_(lane == 0), txl.then_():
    txl.ptx["st.shared.f32"](warp_red.ptr_to([warp]), s)
txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))
txl.assign(row_sum, txl.float32(0.0))
for w in range(NWARP):
    txl.ptx["ld.shared.f32"](t, warp_red.ptr_to([w]))
    txl.assign(row_sum, row_sum + t)
```

Reusing warp_red for both — need a bar.sync between? After all threads read warp_red (broadcast loads), the next write is by lane 0 in the sum reduction after warp_sum shuffle — but there's no barrier between the reads of the max phase and writes of the sum phase. Race: thread in warp 3 might still be reading warp_red[?] from max phase while warp 0's lane 0 writes sum value. Need a bar.sync after the broadcast reads, before the sum-phase store. Add txl.ptx.bar.sync(1, 128) after computing row_max (before pass 2), and similarly after row_sum before pass 3. Or use two different smem arrays and a single barrier before pass 3... Simplest: barriers.

Actually alternative: use warp_red[0..3] for max and warp_red[4..7] for sum — but sum is computed after pass 2 which comes after max reads, still need ordering... The sum-phase store to warp_red[warp] happens after pass 2 (which contains no barrier but loads from smem x_smem written in pass 1 — need barrier between pass 1 and max broadcast? Pass 1 stores x_smem, then warp reduce (no smem), then lane0 stores warp_red, then bar.sync(0). This barrier also orders pass-1 x_smem stores before pass-2 reads. Good.

Then reads of warp_red for row_max (all threads). Then pass 2 reads x_smem (already ordered) and writes exp values to x_smem! Danger: pass 2 writes x_smem[idx] while other threads may still be reading x_smem at other indices — but each thread reads and writes only its own idx, and different threads have different idx. Within pass 2, thread reads x_smem[idx] then writes x_smem[idx] — same location, own location, no race. OK.

Then warp_sum(s), lane0 stores warp_red[warp] — race with late readers of warp_red from max phase! Threads could still be in pass 2 reading... no wait, readers of warp_red for row_max happened before pass 2. All threads must finish reading warp_red[0..3] before any lane-0 writes warp_red[warp] for sum. The lane-0 write happens after that thread finished pass 2. But another thread (e.g., in warp 3) might have finished max broadcast reads... those reads happen before pass 2 for that thread. The write by warp 0 lane 0 happens after warp 0 finished pass 2. But warp 3's broadcast reads happened before warp 3's pass 2 — is warp 3's broadcast read guaranteed to happen before warp 0's sum write? No! Warp 0 could race ahead: finish max broadcast, do pass 2, warp_sum, write warp_red[0] — while warp 3 hasn't even started its max broadcast reads (reading stale/garbage max values). So need bar.sync between max broadcast reads and the sum-phase lane0 store. Insert bar.sync after computing row_max, i.e., before pass 2. That's cheap and also no-op-ish. Then after sum broadcast reads, before pass 3 writes to y (global) — pass 3 reads x_smem exp values; those writes ordered by... pass 3 reads x_smem written in pass 2 by other threads! Thread A writes x_smem[idxA] in pass 2; thread B reads x_smem[idxB] in pass 3 where idxB possibly == idxA? No — in pass 3 each thread reads the same idx it wrote. Own location only. But wait, pass 3 reads x_smem[idx] which the same thread wrote in pass 2. Same thread, program order. No race. And pass-3 doesn't touch warp_red. But is there any subsequent write to warp_red? No, kernel ends. But y stores are global; fine.

Hmm wait, actually pass 2 reads x_smem[idx] written by the same thread in pass 1 too. So the smem staging is per-thread-owned locations throughout — the bar.sync(0) after pass 1 is actually not even needed for x_smem correctness, but needed for warp_red. Keep it.

So barriers: 
1. After pass1 + lane0 store warp_red(max) → bar.sync(0) → broadcast reads → row_max.
2. bar.sync(0) again (to protect warp_red reuse) → pass 2.
3. After pass2 + lane0 store warp_red(sum) → bar.sync(0) → broadcast reads → row_sum → pass 3.

Barrier ID reuse fine.

Actually can I avoid barrier 2 by using separate arrays: warp_red_max[4] and warp_red_sum[4]? Then sum-phase writes to warp_red_sum never conflict with max reads of warp_red_max. Yes! That removes barrier 2. But actually there's another subtle race: pass 3 writes to y global only, no warp_red conflict. So with two arrays, only barriers 1 and 3 needed. Let's do that — cleaner.

Now also think: is the shfl inside `if lane == 0`? No, warp reduce is outside guards, uniform. Good.

Another subtlety: shfl_sync with FULL_MASK requires all 32 lanes of each warp active. block=128, all threads execute the serial loop uniformly and the reduction outside any guard. Good.

NEG_BIG: -3.4028234663852886e+38. Since actual values ~100, fine. Or use first element... simpler with NEG_BIG. But note if n small, many threads have m = NEG_BIG, warp max handles fine.

Actually alternative: initialize m via ld of first element guarded... keep NEG_BIG.

Now division: inv = 1.0 / row_sum. Use txl.assign(inv, txl.float32(1.0) / row_sum). Expression division → div.rn.f32 presumably. Then multiply. Precision fine.

Edge: row_sum could be 0 if all exp underflow? exp(x - max) has at least one element equal to exp(0)=1 (the max element), so sum >= 1. Good.

Now indexing type: idx = (it * NTHREADS + tid) * vec. it is loop var (int32?), tid int32. idx int32. n static int. `idx < n` — comparing int32 expr with Python int; fine.

ptr_to([r, idx]) — r is cta_id (int32). Tensor shape (rows, n) static. Should be fine.

For the non-smem large-n variant, pass 2 loads from global: x.ptr_to([r, idx]) with v4. Pass 3 loads global again and recomputes exp.

Let me now also consider whether to vectorize smem pass-2 loads with v4. Yes when vec==4.

Code structure with Python-level conditionals:

```python
import math
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
NTHREADS = 128
NWARP = 4
LOG2E = 1.4426950408889634
NEG_BIG = -3.4028234663852886e38

LD_G_F32 = "ld.global.f32"
LD_G_V4 = "ld.global.v4.f32"
ST_G_F32 = "st.global.f32"
ST_G_V4 = "st.global.v4.f32"
LD_S_F32 = "ld.shared.f32"
LD_S_V4 = "ld.shared.v4.f32"
ST_S_F32 = "st.shared.f32"
ST_S_V4 = "st.shared.v4.f32"


def make_kernel(rows: int, n: int):
    vec = 4 if n % 4 == 0 else 1
    nvec = n // vec
    iters = (nvec + NTHREADS - 1) // NTHREADS
    use_smem = n * 4 <= 48 * 1024

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        if use_smem:
            x_smem = smem.alloc([n], "float32")
        red_max = smem.alloc([NWARP], "float32")
        red_sum = smem.alloc([NWARP], "float32")

        v = txl.alloc_local([vec], "float32")
        e = txl.alloc_local([vec], "float32")
        m = txl.local_scalar("float32")
        s = txl.local_scalar("float32")
        t = txl.local_scalar("float32")
        row_max = txl.local_scalar("float32")
        row_sum = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")

        def warp_max(acc):
            peer = txl.local_scalar("uint32")
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(acc, ty="uint32"),
                    txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
                txl.ptx["max.f32"](acc, acc, txl.reinterpret(peer, ty="float32"))

        def warp_sum(acc):
            peer = txl.local_scalar("uint32")
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(acc, ty="uint32"),
                    txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
                txl.assign(acc, acc + txl.reinterpret(peer, ty="float32"))

        def load_x(dst, idx):
            if vec == 4:
                txl.ptx[LD_G_V4](dst[0], dst[1], dst[2], dst[3], x.ptr_to([r, idx]))
            else:
                txl.ptx[LD_G_F32](dst[0], x.ptr_to([r, idx]))

        def store_y(idx, dst):
            if vec == 4:
                txl.ptx[ST_G_V4](y.ptr_to([r, idx]), dst[0], dst[1], dst[2], dst[3])
            else:
                txl.ptx[ST_G_F32](y.ptr_to([r, idx]), dst[0])
```

Hmm, dst here is local array; for store_y pass values. OK.

Wait — issue: helpers load_x reference `x` and `r` from enclosing scope — fine.

Pass 1:

```python
        txl.assign(m, txl.float32(NEG_BIG))
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec
            with txl.if_(idx < n), txl.then_():
                load_x(v, idx)
                if use_smem:
                    if vec == 4:
                        txl.ptx[ST_S_V4](x_smem.ptr_to([idx]), v[0], v[1], v[2], v[3])
                    else:
                        txl.ptx[ST_S_F32](x_smem.ptr_to([idx]), v[0])
                for k in range(vec):
                    txl.ptx["max.f32"](m, m, v[k])
```

Max reduction and broadcast:

```python
        warp_max(m)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx[ST_S_F32](red_max.ptr_to([warp]), m)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))
        txl.assign(row_max, txl.float32(NEG_BIG))
        for w in range(NWARP):
            txl.ptx[LD_S_F32](t, red_max.ptr_to([w]))
            txl.ptx["max.f32"](row_max, row_max, t)
```

Pass 2:

```python
        txl.assign(s, txl.float32(0.0))
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec
            with txl.if_(idx < n), txl.then_():
                if use_smem:
                    if vec == 4:
                        txl.ptx[LD_S_V4](v[0], v[1], v[2], v[3], x_smem.ptr_to([idx]))
                    else:
                        txl.ptx[LD_S_F32](v[0], x_smem.ptr_to([idx]))
                else:
                    load_x(v, idx)
                for k in range(vec):
                    txl.assign(e[k], (v[k] - row_max) * txl.float32(LOG2E))
                    txl.ptx["ex2.approx.f32"](e[k], e[k])
                    txl.assign(s, s + e[k])
                if use_smem:
                    if vec == 4:
                        txl.ptx[ST_S_V4](x_smem.ptr_to([idx]), e[0], e[1], e[2], e[3])
                    else:
                        txl.ptx[ST_S_F32](x_smem.ptr_to([idx]), e[0])
```

Hmm — `txl.ptx["ex2.approx.f32"](e[k], e[k])`: dest and source same local. In DPS the backend probably handles it (reads RHS, writes LHS). mov.b32(x_smem... ) in example had distinct. Risk: DPS might require distinct? Unlikely — DPS just means destination passed as first arg. Aliasing should be fine at IR level (it becomes dest = op(src) with src read before write... at trace time, e[k] as src is evaluated to an expression referencing the variable, then assign writes. Should be OK). To be extra safe, use a temp:

```python
tmp = txl.local_scalar("float32")  # declared once outside
...
txl.assign(tmp, (v[k] - row_max) * txl.float32(LOG2E))
txl.ptx["ex2.approx.f32"](e[k], tmp)
```

Do that.

Also alloc_local indexing: e is txl.alloc_local([vec], "float32"); e[k] valid per example (xf[kv]).

Sum reduction:

```python
        warp_sum(s)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx[ST_S_F32](red_sum.ptr_to([warp]), s)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))
        txl.assign(row_sum, txl.float32(0.0))
        for w in range(NWARP):
            txl.ptx[LD_S_F32](t, red_sum.ptr_to([w]))
            txl.assign(row_sum, row_sum + t)
        txl.assign(inv, txl.float32(1.0) / row_sum)
```

Pass 3:

```python
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec
            with txl.if_(idx < n), txl.then_():
                if use_smem:
                    if vec == 4:
                        txl.ptx[LD_S_V4](e[0], e[1], e[2], e[3], x_smem.ptr_to([idx]))
                    else:
                        txl.ptx[LD_S_F32](e[0], x_smem.ptr_to([idx]))
                else:
                    load_x(v, idx)
                    for k in range(vec):
                        txl.assign(tmp, (v[k] - row_max) * txl.float32(LOG2E))
                        txl.ptx["ex2.approx.f32"](e[k], tmp)
                for k in range(vec):
                    txl.assign(e[k], e[k] * inv)
                if vec == 4:
                    txl.ptx[ST_G_V4](y.ptr_to([r, idx]), e[0], e[1], e[2], e[3])
                else:
                    txl.ptx[ST_G_F32](y.ptr_to([r, idx]), e[0])
    return softmax_txl
```

Edge case: iters could be 0? n>=1 → nvec>=1 → iters>=1. Good.

Edge case n not multiple of anything with vec=1: guards handle.

One more check: `idx = (it * NTHREADS + tid) * vec` — `it` is the serial loop variable; in example `col = iteration * 32 + txl.thread_id()`. Good. Multiplication by vec (Python int) fine.

Wait, potential issue with vec=4 and n%4==0 but n < 4? n=0 excluded (n>=1). n=1,2,3 → vec=1. n=4 → vec=4.

Another consideration: smem pool allocations when use_smem False — smem.alloc for red arrays only. Fine.

Potential issue: `txl.ptx["max.f32"]` — does the ptx namespace accept arbitrary instruction strings via __getitem__? Example uses txl.ptx[LD_G_V4] with LD_G_V4 = "ld.global.v4.b32". Yes.

`txl.ptx.bar.sync` attribute form used in example. Keep.

`txl.ptx.shfl_sync.bfly.b32` — keep exact spelling from example.

`txl.reinterpret(acc, ty="uint32")` — example used ty="uint32" string. Good.

For local_scalar in warp helpers: acc is passed as local_scalar; txl.reinterpret(acc, ...) — acc is a scalar var; example reinterpreted acc[0] (array element). reinterpret of a local_scalar should work similarly. Hmm, local_scalar usage: txl.assign(goff, offset), goff used as value. reinterpret(goff, ty="uint32")? Should be fine since it's an expression.

Actually wait — in the example, `ss` is alloc_local([1], "float32") and used as ss[0]. Maybe local_scalar behaves like a scalar variable directly. txl.local_scalar("float32") then txl.assign(rms, ...), used as rms in expressions. Yes per example (rms, row, goff). reinterpret on it should be fine.

But shfl needs a uint32 peer local: example used txl.local_scalar("uint64") for pk. So txl.local_scalar("uint32") fine.

One concern: `txl.ptx["max.f32"](acc, acc, txl.reinterpret(peer, ty="float32"))` — 3-operand DPS. mov.b32 was 2-operand; mul.f32 3-operand like example's? Example used expression arithmetic (ss[0] + xf[kv]*xf[kv]) rather than ptx mul. And shfl_sync.bfly.b32 took 5 args. txl.ptx instructions presumably support their natural operand lists. max.f32 d, a, b — should be fine. Alternatively use txl.Select? Hmm, txl.Select(cond, a, b) exists per docs (used with descriptors). Could do txl.assign(acc, txl.Select(acc > p, acc, p)). But Select on floats returns... it's generic. I'll use ptx max.f32 — it's a standard PTX instruction and txl.ptx is "PTX spelled through txl.ptx".

Hmm, but there's a risk with ftz: max.f32 vs max.ftz.f32 — non-ftz fine.

ex2.approx.f32: 2 operands (d, a). DPS with mov.b32 precedent (2 operands). OK.

Now, `txl.float32(NEG_BIG)` — example used txl.float32(0), txl.float32(eps). Passing a large negative float fine.

One more: the `with txl.if_(idx < n)` — idx is int32 expression, n Python int. In the verified example: `with txl.if_(col < cols)`. Good.

Alignment of x_smem for v4: smem pool base presumably 16B-aligned (usually 128B). x_smem is first allocation → offset 0. idx multiple of 4 elements → 16B aligned. Good. When !use_smem, red arrays at offsets 0 and 16.

Wait, when use_smem and vec==4: x_smem [n], then red_max at offset n*4 bytes — n%4==0 so 16B aligned anyway, but red arrays only need 4B. Fine.

Also alignment for global v4 loads: x base pointer 16B aligned (torch allocations 256B), row offset r*n*4 with n%4==0 → multiple of 16. idx multiple of 4 → ptr aligned. y same. 

Now also consider: should I add `min_blocks_per_sm`? Not necessary.

Double-check the bar.sync count operand: txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(nthreads)) — matches example.

Potential correctness issue: pass 2 for !use_smem loads global again — fine.

What about `e` array when vec==1: alloc_local([1], "float32"), e[0]. Fine.

Now, one thing about txl.serial(iters) where iters is a Python int — example: `with txl.serial(n_tiles) as ki` with n_tiles Python int. Good.

Let me also double check ptr_to for 2D tensor: x.ptr_to([r, col]) used in scale2 example. Good. idx expression: `(it * NTHREADS + tid) * vec` — mixing loop var with ints. In example: `col = iteration * 32 + txl.thread_id()`... wait that's `iteration * 32`, where 32 is Python int, thread_id() int32. Fine.

Hmm, `(it * NTHREADS + tid) * vec`: for vec=1 it's it*128 + tid. Good.

One more potential problem: in `load_x` helper I use parameter name dst and index dst[0..3]. alloc_local arrays support indexing. Good.

Also: should the exp use precise exp? ex2.approx error ~2 ulp; sum relative error negligible. Fine.

Numerical stability with x ~ 100: x - max in [-inf, 0]; (x-max)*log2e could be very negative → ex2.approx.f32 underflows to 0. Good. Positive? x - max <= 0 always. Good.

Also consider potential issue: ex2.approx.f32 might have issues with denormals/ftz — doesn't matter.

Now, "one CTA (128 threads) per row" — grid=rows, block=128. Yes.

Final code assembly. Also note: don't enable `from __future__ import annotations`. Use annotations as strings? The interface given: `x: txl.Tensor((rows, n), "float32")` — rows, n are Python ints captured by closure. Good.

Also must return softmax_txl.

Let me also reconsider: should pass 2's smem store and pass 3's smem read be guarded consistently — yes both under idx<n.

One thing: after pass 1, threads with idx >= n have m = NEG_BIG, participate in warp_max fine.

Also for very small n (like n=1), only warp 0 lane 0 loaded; warps 1-3 have m=NEG_BIG; red_max[1..3] = NEG_BIG; row_max correct.

Sum: s=0 for out-of-range threads; correct.

I think the kernel is solid. Write final module.

One more consideration on `txl.ptx["st.shared.f32"]` vs example ST_S_F32 = "st.shared.f32". Good.

Also the module-level constants: fine.

Should I worry that `txl.ptx[ST_G_V4]` for f32 vector: "st.global.v4.f32" is valid PTX. Yes.

Also ld.global.v4.f32 requires the pointer operand; DPS: txl.ptx[LD_G_V4](v[0], v[1], v[2], v[3], x.ptr_to([r, idx])). Matches example pattern for v4.b32.

Edge: `use_smem` threshold 48KB: A100 has 164KB configurable smem per CTA, but 48KB is the static limit without opt-in attribute. Keep 48KB → n <= 12288 uses smem.

Actually, could also add `kernel_attrs` for dynamic smem... not needed.

Write final answer.

## content
```python
# Row-wise softmax (sm_80): one 128-thread CTA per row.
#   Small/medium rows: stage the row in shared memory (1 global read, 1 global write).
#   Large rows (> 48KB): three streaming global passes, no row staging.
# Numerically stable: subtract the row max before ex2.

import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
NTHREADS = 128
NWARP = 4
LOG2E = 1.4426950408889634
NEG_BIG = -3.4028234663852886e38
SMEM_BUDGET = 48 * 1024

LD_G_F32 = "ld.global.f32"
LD_G_V4 = "ld.global.v4.f32"
ST_G_F32 = "st.global.f32"
ST_G_V4 = "st.global.v4.f32"
LD_S_F32 = "ld.shared.f32"
LD_S_V4 = "ld.shared.v4.f32"
ST_S_F32 = "st.shared.f32"
ST_S_V4 = "st.shared.v4.f32"
MAX_F32 = "max.f32"
EX2_F32 = "ex2.approx.f32"


def make_kernel(rows: int, n: int):
    # vec=4 only when it divides n exactly, so every guarded access is a
    # full, 16B-aligned vector and the tail needs no element masks.
    vec = 4 if n % 4 == 0 else 1
    nvec = n // vec
    iters = (nvec + NTHREADS - 1) // NTHREADS
    use_smem = n * 4 <= SMEM_BUDGET

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=rows, block=128),
            backend_config={"cuda": {"arch": "sm_80"}},
        )

        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        if use_smem:
            x_smem = smem.alloc([n], "float32")
        red_max = smem.alloc([NWARP], "float32")
        red_sum = smem.alloc([NWARP], "float32")

        v = txl.alloc_local([vec], "float32")  # raw row values
        e = txl.alloc_local([vec], "float32")  # exp values
        m = txl.local_scalar("float32")  # thread-local max
        s = txl.local_scalar("float32")  # thread-local sum
        t = txl.local_scalar("float32")  # scratch
        tmp = txl.local_scalar("float32")  # scratch
        row_max = txl.local_scalar("float32")
        row_sum = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")

        def warp_max(acc):
            peer = txl.local_scalar("uint32")
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(acc, ty="uint32"),
                    txl.uint32(delta),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.ptx[MAX_F32](acc, acc, txl.reinterpret(peer, ty="float32"))

        def warp_sum(acc):
            peer = txl.local_scalar("uint32")
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(acc, ty="uint32"),
                    txl.uint32(delta),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.assign(acc, acc + txl.reinterpret(peer, ty="float32"))

        def load_global(dst, idx):
            if vec == 4:
                txl.ptx[LD_G_V4](dst[0], dst[1], dst[2], dst[3], x.ptr_to([r, idx]))
            else:
                txl.ptx[LD_G_F32](dst[0], x.ptr_to([r, idx]))

        # ---- pass 1: read the row, stage it, track the thread-local max ----
        txl.assign(m, txl.float32(NEG_BIG))
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec
            with txl.if_(idx < n), txl.then_():
                load_global(v, idx)
                if use_smem:
                    if vec == 4:
                        txl.ptx[ST_S_V4](x_smem.ptr_to([idx]), v[0], v[1], v[2], v[3])
                    else:
                        txl.ptx[ST_S_F32](x_smem.ptr_to([idx]), v[0])
                for k in range(vec):
                    txl.ptx[MAX_F32](m, m, v[k])

        # ---- CTA max: warp butterfly, warp leaders to smem, broadcast back --
        warp_max(m)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx[ST_S_F32](red_max.ptr_to([warp]), m)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))
        txl.assign(row_max, txl.float32(NEG_BIG))
        for w in range(NWARP):
            txl.ptx[LD_S_F32](t, red_max.ptr_to([w]))
            txl.ptx[MAX_F32](row_max, row_max, t)

        # ---- pass 2: exp(x - row_max), accumulate the sum ------------------
        txl.assign(s, txl.float32(0.0))
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec
            with txl.if_(idx < n), txl.then_():
                if use_smem:
                    if vec == 4:
                        txl.ptx[LD_S_V4](v[0], v[1], v[2], v[3], x_smem.ptr_to([idx]))
                    else:
                        txl.ptx[LD_S_F32](v[0], x_smem.ptr_to([idx]))
                else:
                    load_global(v, idx)
                for k in range(vec):
                    txl.assign(tmp, (v[k] - row_max) * txl.float32(LOG2E))
                    txl.ptx[EX2_F32](e[k], tmp)
                    txl.assign(s, s + e[k])
                if use_smem:
                    if vec == 4:
                        txl.ptx[ST_S_V4](x_smem.ptr_to([idx]), e[0], e[1], e[2], e[3])
                    else:
                        txl.ptx[ST_S_F32](x_smem.ptr_to([idx]), e[0])

        # ---- CTA sum: same pattern (separate smem slots, one barrier) ------
        warp_sum(s)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx[ST_S_F32](red_sum.ptr_to([warp]), s)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(NTHREADS))
        txl.assign(row_sum, txl.float32(0.0))
        for w in range(NWARP):
            txl.ptx[LD_S_F32](t, red_sum.ptr_to([w]))
            txl.assign(row_sum, row_sum + t)
        txl.assign(inv, txl.float32(1.0) / row_sum)

        # ---- pass 3: normalize and write out -------------------------------
        with txl.serial(iters) as it:
            idx = (it * NTHREADS + tid) * vec
            with txl.if_(idx < n), txl.then_():
                if use_smem:
                    if vec == 4:
                        txl.ptx[LD_S_V4](e[0], e[1], e[2], e[3], x_smem.ptr_to([idx]))
                    else:
                        txl.ptx[LD_S_F32](e[0], x_smem.ptr_to([idx]))
                else:
                    load_global(v, idx)
                    for k in range(vec):
                        txl.assign(tmp, (v[k] - row_max) * txl.float32(LOG2E))
                        txl.ptx[EX2_F32](e[k], tmp)
                for k in range(vec):
                    txl.assign(e[k], e[k] * inv)
                if vec == 4:
                    txl.ptx[ST_G_V4](y.ptr_to([r, idx]), e[0], e[1], e[2], e[3])
                else:
                    txl.ptx[ST_G_F32](y.ptr_to([r, idx]), e[0])

    return softmax_txl
```
