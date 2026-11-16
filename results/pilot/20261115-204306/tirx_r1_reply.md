# tirx round 1

model=glm-5.3 in=7736 out=23755 cost=0.11114 latency=726.0s finish=stop effort=high usage={"completion_tokens": 23755, "completion_tokens_details": null, "prompt_tokens": 7736, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 7680, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 21661, "total_tokens": 31491}

## reasoning
We need to write a softmax kernel for sm_80, one CTA (128 threads) per row, static shapes rows, n. Requirements: correct for any n >= 1, numerically stable, high performance.

Design: For performance, we'd like vectorized loads/stores when n is a multiple of 4, plus handling tail. Since shapes are static (one kernel per (rows, n)), we can choose vec at trace time: vec = 4 if n % 4 == 0 else 1. But we could also do main loop of 4-wide vectors with a scalar tail: process floor(n/4)*4 with v4, then tail of n%4 scalars. That works for any n. Use ld.global.v4.b32 / st.global.v4.b32 (16 bytes).

Approach:
- Read x into shared memory (as float32) while computing max (first pass), then block reduce max, then second pass over smem: compute exp(x - max), accumulate sum, block reduce sum, then third pass: y = exp/scale stored. Or two passes: pass1 read to smem + max reduce; pass2 read smem, compute e = exp(x-max), store e back to smem, accumulate sum; reduce; pass3 read smem, divide, store to global. Alternatively store exp values in registers per thread instead of smem for pass3 — that avoids smem pass 3. But threads may hold variable number of vector registers... with static n, we can compute per-thread max registers: nvec = ceil(n/4)... Actually simpler: keep in smem; smem traffic is cheap relative to global.

Actually, to reduce work: online softmax single pass with rescaling? With static shapes, two passes over global memory might be avoided by staging in smem. n could be large (e.g., 8192 floats = 32KB, fits in A100 smem 164KB... but 48KB default without opt-in; txl smem_pool might handle that). Hmm, if n is very large (like 100K), smem won't fit. Task says "n is often not a multiple of 128" — typical softmax rows like 256..8192. Should we handle arbitrary n with a fallback that doesn't use smem staging? To be safe: if n > some threshold (e.g., smem budget like 32KB / 8 elements?), use a version reading from global twice. Since one kernel per (rows, n), we can pick code paths in Python at trace time.

Let me design two code paths:

Path A (n floats fit in smem, say n <= 24576 (~96KB? A100 has 164KB max per SM, 163KB max per block opt-in; default static smem limit 48KB without attribute). txl smem_pool — does it set dynamic smem attribute? Unknown. Safer threshold: n <= 8192 (32KB) — within default 48KB static limit? Static shared memory over 48KB fails to compile without cudaFuncAttributeMaxDynamicSharedMemorySize. txl smem_pool likely allocates dynamic smem and maybe handles opt-in. To be safe, use n*4 <= 48KB → n <= 12288. Let's set threshold n <= 12288 for smem path; else global-rescan path.

Actually maybe simpler: always use a global two-pass approach without smem staging? That reads x twice from global (L2 likely caches second pass, but L2 hit still slower than smem). For performance, smem staging is better. I'll implement both.

But maybe keep it simpler: one code path with smem staging when it fits, else fallback that recomputes exp from global reads (three global passes: max, sum, write — reading twice). Let's write both.

Kernel structure (smem path, vec = 4 when n % 4 == 0? Better: handle any n with vec4 main + scalar tail, so single code path for any n):

Pass 1: for each chunk index i in range(ntiles) where each thread handles vec=4 elements per iteration: idx = it*128*4 + tid*4. If idx + 4 <= n: vector load x[r, idx:idx+4], compute m = max of 4, local max update, store v4 to smem. Else handle tail: scalar loop over remaining elements for the thread that owns them (guard idx < n). Hmm — with strided layout thread t handles indices [it*512 + t*4, +4). Tail: elements from nv4*4 to n-1, up to 3 elements, owned by thread(s). Simpler: main vector loop over nv4 = n // 4 full vectors, then tail loop: scalar idx over range(nv4*4, n) processed by threads with tid == idx - nv4*4? Or just let thread 0..(n%4 - 1) handle one each: idx = nv4*4 + tid, if tid < n%4.

For max computation, use local scalar m, init to -inf. Use txl ops: max via txl.max? Does txl have max/min? Native TIRx expressions: txl.max likely exists (TIR has max). The RMSNorm example uses `min`, `+`, `*`, `/`, rsqrt via txl.rsqrt. I'd guess txl.max(a,b) exists as native PrimFunc-free op. To be safe, could use `txl.Select(a > b, a, b)`? The guide mentions txl.Select for descriptors but it's a general expression constructor. Hmm, using Select as max is fine: `txl.Select(a >= b, a, b)`. But is Select available for general scalars? "an FFI-backed expression constructor such as txl.Select accepts either wrapper directly" — Select is native. I'll use txl.max — risky. TIR definitely has Max/Min; tirx-lite likely exposes txl.max/txl.min. Hmm. The guide says "Primitive expressions follow the native signatures". I think txl.max and txl.min exist in TIRx script namespace (like tvm.script's T.max). tirx is a fork of TVM-ish. tvm_tirx.script has T.max likely. txl namespace probably re-exports. To reduce risk, I can implement max via Select... but Select arms with descriptors; general use fine.

Actually safer alternative: since f32 ops, use txl.ptx? There's no scalar max.f32 PTX instruction (there is max.f32 actually! PTX has max.f32). Yes, PTX `max.f32 d, a, b` exists. So I can use txl.ptx.max.f32(dst, a, b) in DPS form like shfl example. That's safest: `txl.ptx.max.f32(m, m, v)`. PTX max.f32 with NaN handling — fine for softmax (assume no NaN). Also min.f32 exists. Good.

Also exp: PTX has ex2.approx.f32 (2^x). exp(x) = ex2.approx(x * log2(e)). Use txl.ptx.ex2.approx.f32(dst, src). Accuracy: ex2.approx has ~2 ulp; fine for softmax. The task says numerically stable for large |x| — stability means subtract max; approx exp is fine (max relative error ~2^-22). Use ex2.approx.f32 with multiply by 1.4426950408889634. Alternatively txl.exp? Guide shows txl.rsqrt exists, so txl.exp likely exists. But ex2.approx is faster and single instruction. I'll use ex2.approx.

Warp reduction: shfl.sync.bfly.b32 like example, with max: need shfl then max.f32. Use txl.ptx.shfl_sync.bfly.b32(peer, reinterpret(m), delta, 31, FULL_MASK) then max.f32.

Block reduction with 4 warps: warp reduce, lane0 writes to smem, bar.sync, warp 0 lanes 0..3 read, reduce, broadcast via smem + bar.sync, then all threads read.

Guard tail elements: initialize m to -inf (0xFF800000). Use txl.float32("-inf")? Does that work? Could construct via reinterpret: txl.reinterpret(txl.uint32(0xFF800000), ty="float32")... guide shows txl.reinterpret(value, ty=txl.uint32). So txl.reinterpret(txl.uint32(0xFF800000), ty="float32") should give -inf. Or use txl.float32(float("-inf")) — Python float('-inf') maybe acceptable. I'll use reinterpret to be safe. Actually, since rows exist and n>=1, every thread participates in reduction; init m = -inf then max over loaded values; threads with no elements keep -inf, fine after block max.

Actually with vec4 + tail, all elements handled. m starts at -inf.

Pass 1 detail:

nthreads = 128, vec = 4.
nfull = n // 4  (number of full vec4 groups)
ntiles = ceil(nfull / 128)
tail elements: t0 = nfull*4, rem = n - t0.

Pass 1:
m = -inf
for it in serial(ntiles):
    base = (it*128 + tid)*4
    if base + 4 <= n:  # base < t0 equivalent since base multiple of 4: base < t0
        ld.global.v4.b32 words? For float32 vec4, could use ld.global.v4.f32 directly with float locals: `txl.ptx.ld.global.v4.f32(a,b,c,d, ptr)`. RMSNorm used ld.global.v4.b32 for f16x2 packed words. For f32, ld.global.v4.f32 exists in PTX. Does txl.ptx support arbitrary instruction names via attribute access (txl.ptx.ld.global_.f32 shown, and txl.ptx[LD_G_V4])? Yes, indexing with strings works. I'll use txl.ptx["ld.global.v4.f32"](v0,v1,v2,v3, ptr). And st.shared.v4.f32 as in example (LD_S_V4 = "ld.shared.v4.f32" so shared vector f32 supported).
        max update each.
        st.shared.v4.f32.
    else: (only last tile) handle scalars:
        with txl.if_(base < n): scalar loop? Elements base..min(base+4,n). Since base is multiple of 4 and base >= t0, remaining count = n - base for this thread (threads beyond have base >= n). Use for k in range(4) unrolled? Trace-time Python for with txl.if_(base+k < n): scalar ld, max, st.shared.f32. Fine.
Also tail after nfull*4: threads with it = ntiles-1... wait tail elements t0..n-1: t0 is a multiple of 4, belongs to thread with base==t0 in last tile (if nfull % 128 == 0, base t0 belongs to tile ntiles-1, tid 0... careful: if nfull % 128 == 0, then ntiles = nfull/128 and last tile's bases start at (ntiles-1)*512; thread tid has base = t0 - 512 + tid*4... no: base = (it*128+tid)*4. For it=ntiles-1, tid=0..127, base covers [(ntiles-1)*512, ntiles*512). t0 = nfull*4 = ntiles*512 if nfull%128==0. Then t0 >= ntiles*512, i.e., tail elements are beyond the last tile's range! Bug. Handle: tail handled by an extra scalar segment after the vector loop, indexed tid < rem, idx = t0 + tid. That's cleaner:

Pass1:
for it in serial(ntiles): vector part guarded base + 4 <= n (equivalently base < t0, but since base multiple of 4, base < t0 ⟺ base+4 <= n... if base < t0 then base+4 <= t0 <= n yes; if base >= t0 then base >= t0 and base+4 > n? base >= t0, n < base+4? n - t0 = rem < 4, base >= t0, base is multiple of 4 ≥ t0 (t0 multiple of 4) so base >= t0 and n <= t0+3 < base+4 when base==t0; if base > t0 then base >= t0+4 > n. Yes equivalent.)

Hmm but when base >= t0 in last tile, the guard fails and thread skips — but tail elements t0..n-1 belong to thread with base == t0 (if that thread exists in last tile). With the separate tail segment (tid < rem, idx = t0 + tid), all rem elements handled by threads 0..rem-1 regardless. But wait: if rem>0 and nfull%128 != 0, the thread with base == t0 is in the last tile (tid = nfull%128), and it will skip the vector guard and handle tail via separate segment. Good. And if nfull%128==0, tail handled by tid 0..rem-1 with idx t0+tid. Good.

So structure:

# vector pass
with txl.serial(ntiles) as it:
    base = (it * 128 + tid) * 4
    with txl.if_(base + 4 <= n), txl.then_():
        ld v4 from x.ptr_to([r, base])  — wait, ptr_to takes indices [row, col]; for vector, pass pointer to first element. x.ptr_to([r, base]) then ld.global.v4.f32 loads 4 consecutive. OK.
        max updates
        st.shared.v4.f32(smem.ptr_to([base]), v0..v3)
# tail
rem = n - t0 (trace-time constant)
for k in range(rem):  # 0..3, Python unrolled
    idx = t0 + k
    with txl.if_(tid == k):  # or tid < rem handled per k
        ld.global.f32; max; st.shared.f32

Simplify: for k in range(rem): if tid == k: handle idx=t0+k. Or single guard tid < rem with a loop over k in range(rem) each guarded tid==k. Use the simple per-k guard.

Alternatively use serial loop? rem is trace-time constant, Python unroll is fine.

Then block max reduce. Then pass 2:

s = 0.0 (sum accumulator)
with txl.serial(ntiles) as it:
    base = (it*128 + tid)*4
    with txl.if_(base + 4 <= n):
        ld.shared.v4.f32 from smem
        e_i = ex2((v_i - mx) * log2e)
        s += e0+e1+e2+e3
        st.shared.v4.f32 back
tail: for k in range(rem): if tid==k: v=ld.shared.f32; e=ex2(...); s+=e; st.shared.f32

block sum reduce. inv = 1/s (use txl division or rcp.approx? use div.rn? Use txl float division: `txl.float32(1.0) / s` — native expression. Fine.)

Pass 3:
with txl.serial(ntiles):
    ld.shared.v4, multiply by inv, st.global.v4
tail similarly.

bar.sync needed between pass1 stores and pass2 loads (after block reduce there's already a bar.sync from reduction). Between pass2 smem stores and pass3 smem loads: the block-sum reduction contains bar.sync, which covers it. Ensure ordering: after pass2 stores, we do warp reduce, lane0 st shared, bar.sync, warp0 reads... those reuse reduction smem — but pass3 reads data smem after the final bar.sync of reduction. The reduction's bar.sync guarantees all threads' pass-2 smem stores are visible. Yes: warp reduce (shfl, no smem), lane0 writes red_smem[warp], bar.sync — at this barrier all threads have completed their smem data stores too (program order before barrier... memory ordering: bar.sync also acts as memory barrier for shared memory? bar.sync guarantees shared memory accesses before it are visible after. Yes, bar.sync orders shared memory). Then warp0 reduces, writes broadcast, bar.sync again, all read total. Then pass3 reads data smem — safe.

Block reduction helper (for both max and sum):

red = smem.alloc([4], "float32")

def block_reduce(val, combine):  # combine via ptx instruction name
    warp reduce via shfl butterfly with combine op
    if lane==0: st.shared red[warp]
    bar.sync
    if warp==0:
        if lane<4: v = ld.shared red[lane] else v = identity
        warp reduce
        if lane==0: st.shared red[0]
    bar.sync
    result = ld.shared red[0]

Identity for max: -inf; for sum: 0.0.

But the combine op differs (max.f32 vs add.f32). Write two helpers or parametrize with a Python closure taking dst,a,b and calling appropriate ptx. In traced code, Python function fine.

bar.sync: txl.ptx.bar.sync(uint32(0), uint32(128)) like example's cta_sync.

Warp butterfly for f32:

def warp_reduce(val, op):  # op: function(dst, a, b)
    peer = txl.local_scalar("uint32")
    for delta in (16,8,4,2,1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(val, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF))
        p = txl.reinterpret(peer, ty="float32")
        op(val, val, p)  # DPS: txl.ptx.max.f32(val, val, p)

Wait val is local_scalar("float32") — can it be a destination of ptx instruction? In example, `txl.ptx.shfl_sync.bfly.b32(peer, ...)` with peer = txl.local_scalar("uint32") as dest. And `txl.ptx.mov.b32(dst[2*i], ...)` with alloc_local array element. So local_scalar works as DPS dest. And max.f32 DPS: txl.ptx.max.f32(val, val, p). Hmm, does txl.ptx support max.f32 via attribute access? txl.ptx.ld.global_.f32, txl.ptx.st.global_.f32, txl.ptx.bar.sync, txl.ptx.shfl_sync.bfly.b32, txl.ptx.mov.b32, txl.ptx[NAME]. The naming with dots via attribute access: txl.ptx.max.f32 — "max" attr then "f32"? Attribute access with dots should work the same as ld.global_.f32 (ld → global_ → f32). Hmm "global_" has underscore because Python identifiers can't have "ld.global.f32"? Actually txl.ptx.ld.global_.f32: attribute `global_` maps to "global"? Presumably the ptx namespace converts attribute chains to dotted instruction names, with trailing underscores stripped or mapped. Then txl.ptx.max.f32 would be instruction "max.f32". That seems plausible. But to be extra safe, use string indexing: txl.ptx["max.f32"](dst, a, b) — the example used txl.ptx[LD_G_V4] with LD_G_V4 = "ld.global.v4.b32". String indexing definitely works. I'll use string constants for all non-trivial instructions.

Instructions needed:
- "ld.global.v4.f32" (4 dests, ptr) — wait, does ld with vector take multiple dests in DPS form? Example: txl.ptx[LD_G_V4](iw[0], iw[1], iw[2], iw[3], inp.ptr_to(...)) — yes, multiple dests.
- "ld.global.f32"
- "st.global.v4.f32", "st.global.f32"
- "ld.shared.v4.f32", "ld.shared.f32", "st.shared.v4.f32", "st.shared.f32" — example confirms shared v4 f32.
- "shfl.sync.bfly.b32" via txl.ptx.shfl_sync.bfly.b32 (confirmed in example).
- "max.f32", "add.f32", "ex2.approx.f32", "bar.sync" (example: txl.ptx.bar.sync).

ex2.approx.f32 DPS: txl.ptx["ex2.approx.f32"](dst, src).

Shared pointer: smem tensor ptr_to([idx]) gives shared-space pointer? In RMSNorm example, x_smem.ptr_to([st]) used with ld.shared. Yes.

Now indexing x.ptr_to([r, base]): r = txl.cta_id() (int32?), base int32 expression. Row-major 2D tensor with static shape (rows, n) — ptr_to handles addressing. Good. Note example used gidx with int64 promotion because axis was int64; our shape is static Python ints, so indices stay int32 presumably. Fine.

Now the fallback path for large n (smem too big): threshold. Let's decide: use smem staging if n * 4 <= 48*1024, i.e., n <= 12288. Actually also consider occupancy: 32KB smem per block limits to fewer blocks; typical softmax rows ≤ 8192. Alternative: registers-only approach: each thread holds its elements in registers (n/128 floats per thread). For n up to e.g. 8192, that's 64 floats per thread = 64 registers — feasible! Register-resident avoids smem entirely and pass 3 reads registers. But register arrays must be static-size locals; with n/128 up to 64, alloc_local([per_thread]) with dynamic indexing problem: indexing local arrays with runtime index (it loop) — local arrays indexed by loop var may not be supported well (would need unrolled loops). Since ntiles is trace-time constant, we could fully unroll with Python loops: for it in range(ntiles): ... each with its own locals. That generates lots of code for large n (e.g., n=8192 → 64 iterations × few ops — fine). But unrolled Python loops with guards per element: 8192/128=64 iterations per thread, each with ld.v4 — code size okay.

But simpler and robust: smem staging. smem read/write is cheap. I'll do smem path for n <= 12288, and a global-reread path for larger n. Actually for larger n, we can still avoid smem staging by re-reading global twice (pass1: max; pass2: exp+sum, write exp values... we need exp values again in pass3 — recompute exp or store to smem? For huge n, store exp to global y then pass3 rescale y in place? y = y * inv — read/write global again. That's 3 global passes: read (max), read (sum) + we could compute sum while... Let's do: pass1 max (read global). pass2: read global, e=exp(x-mx), sum, store e to y (global). pass3: read y, multiply inv, store y. Total global traffic: 2R + 1RW + 1R + 1W = fine for huge rows. This is the classic 3-pass approach for very long rows. Correct for any n.

Actually, could also chunk with online rescaling, but 3-pass is fine and simple.

Both paths share block reduction helpers. Let me structure the module:

```python
import tirx_kernels.tirx_lite as txl

BLOCK = 128
FULL_MASK = 0xFFFFFFFF
LOG2E = 1.4426950408889634
NEG_INF_BITS = 0xFF800000
SMEM_MAX_ELEMS = 12288  # 48 KB of float32
```

Helpers (traced, defined inside make_kernel or module level using txl only — they're called during trace so they can be module-level functions; they need txl only). Module-level helper functions are fine (like RMSNorm's inner defs). I'll define them inside the kernel body or as module-level functions taking needed values. Since they emit IR when called during trace, module-level is fine.

Let me write:

```python
def _warp_reduce(val, combine):
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer,
            txl.reinterpret(val, ty="uint32"),
            txl.uint32(delta),
            txl.uint32(31),
            txl.uint32(FULL_MASK),
        )
        combine(val, val, txl.reinterpret(peer, ty="float32"))
```

Wait: val is local_scalar("float32"); txl.reinterpret(val, ty="uint32") — guide shows txl.reinterpret(value, ty=txl.uint32) with enum; example used ty="uint32" string: `txl.reinterpret(acc[0], ty="uint32")` and `txl.reinterpret(peer, ty="float32")`. Good.

combine as closure: e.g.

```python
def fmax(dst, a, b):
    txl.ptx["max.f32"](dst, a, b)

def fadd(dst, a, b):
    txl.ptx["add.f32"](dst, a, b)
```

Block reduce:

```python
def _block_reduce(val, red_smem, combine, identity):
    _warp_reduce(val, combine)
    lane = txl.thread_id() & 31  # or lane_id
    warp = txl.thread_id() >> 5
    with txl.if_(lane == 0), txl.then_():
        txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), val)
    cta_sync(0)
    with txl.if_(warp == 0), txl.then_():
        v = txl.local_scalar("float32")
        with txl.if_(lane < 4), txl.then_():
            txl.ptx["ld.shared.f32"](v, red_smem.ptr_to([lane]))
        with txl.else_():
            txl.assign(v, txl.float32(identity))
        _warp_reduce(v, combine)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx["st.shared.f32"](red_smem.ptr_to([0]), v)
    cta_sync(0)
    out = txl.local_scalar("float32")
    txl.ptx["ld.shared.f32"](out, red_smem.ptr_to([0]))
    return out
```

Hmm, identity as float: for max, -inf. Passing Python float("-inf") to txl.float32 — may or may not work. Use reinterpret: txl.reinterpret(txl.uint32(NEG_INF_BITS), ty="float32"). I'll pass a value expression instead of a float: identity param is an expression. For sum: txl.float32(0.0). For max: reinterpret bits.

Note val is local_scalar; reading it after as value fine. Also each call creates fresh locals — fine.

cta_sync:

```python
def _cta_sync():
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
```

Now the main kernel:

```python
def make_kernel(rows: int, n: int):
    vec = 4
    nfull = n // 4
    ntiles = (nfull + BLOCK - 1) // BLOCK
    rem = n - nfull * 4
    use_smem = n <= SMEM_MAX_ELEMS

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()

        smem = txl.smem_pool()
        red = smem.alloc([4], "float32")
        xs = None
        if use_smem:
            xs = smem.alloc([n], "float32")

        m = txl.local_scalar("float32")
        txl.assign(m, NEG_INF_F32)
        ...
```

Wait — Python `if use_smem:` branching at trace time is fine (ordinary Python control flow executes while constructing).

Pass 1 (smem path):

```python
v0 = txl.local_scalar("float32")  # or alloc_local([1])
```

Actually simpler to use alloc_local arrays like example: `xf = txl.alloc_local([4], "float32")` and index xf[0..3]. Example used alloc_local with ptr-style indexing. Use alloc_local([4], "float32").

```python
xf = txl.alloc_local([4], "float32")
```

Pass 1:

```python
with txl.serial(ntiles) as it:
    base = (it * BLOCK + tid) * 4
    with txl.if_(base + 4 <= n), txl.then_():
        txl.ptx["ld.global.v4.f32"](xf[0], xf[1], xf[2], xf[3], x.ptr_to([r, base]))
        for k in range(4):
            txl.ptx["max.f32"](m, m, xf[k])
        txl.ptx["st.shared.v4.f32"](xs.ptr_to([base]), xf[0], xf[1], xf[2], xf[3])
for k in range(rem):
    idx = nfull * 4 + k
    with txl.if_(tid == k), txl.then_():
        txl.ptx["ld.global.f32"](xf[0], x.ptr_to([r, idx]))
        txl.ptx["max.f32"](m, m, xf[0])
        txl.ptx["st.shared.f32"](xs.ptr_to([idx]), xf[0])
```

Edge: n small, e.g., n=1: nfull=0, ntiles=0, rem=1. serial(0) — is serial(0) allowed? Should be (loop with 0 trips). Hmm, `txl.serial(0)` — extent 0 might be fine or might be disallowed. To be safe: if ntiles > 0 emit loop. Python guard.

Also note base is expression (it*BLOCK + tid)*4; comparing base + 4 <= n — n is Python int constant, so comparison becomes constant-folded partially: base+4 <= n. Fine.

Block max:

```python
mx = _block_reduce(m, red, fmax, NEG_INF_F32)
```

Wait, but _block_reduce takes val local and returns local out. But m's value gets modified in place by warp reduce (that's fine).

Note: block reduce uses bar.sync — also serves as the barrier making pass-1 smem stores visible. But careful: the reduction's first bar.sync is after lane0 writes red[warp]; pass-1 data stores happen before that in program order per thread, and bar.sync orders them. Good.

Pass 2:

```python
s = txl.local_scalar("float32")
txl.assign(s, txl.float32(0.0))
log2e = txl.float32(LOG2E)
with txl.serial(ntiles) as it:
    base = (it * BLOCK + tid) * 4
    with txl.if_(base + 4 <= n), txl.then_():
        txl.ptx["ld.shared.v4.f32"](xf[0], ..., xs.ptr_to([base]))
        for k in range(4):
            txl.ptx["ex2.approx.f32"](xf[k], (xf[k] - mx) * log2e)
        txl.ptx["add.f32"](s, s, xf[0]) ... for k
        txl.ptx["st.shared.v4.f32"](xs.ptr_to([base]), xf[0..3])
for k in range(rem):
    idx = nfull*4 + k
    with txl.if_(tid == k), txl.then_():
        txl.ptx["ld.shared.f32"](xf[0], xs.ptr_to([idx]))
        txl.ptx["ex2.approx.f32"](xf[0], (xf[0] - mx) * log2e)
        txl.ptx["add.f32"](s, s, xf[0])
        txl.ptx["st.shared.f32"](xs.ptr_to([idx]), xf[0])
```

Block sum:

```python
tot = _block_reduce(s, red, fadd, txl.float32(0.0))
inv = txl.local_scalar("float32")
txl.assign(inv, txl.float32(1.0) / tot)
```

Division: `txl.float32(1.0) / tot` — native expression division. Should work (example used ss[0] / float32(hidden_size)). Or rcp.approx.f32: txl.ptx["rcp.approx.f32"](inv, tot). Use rcp.approx for speed — accuracy 1 ulp-ish, fine. I'll use div via normal expression for accuracy... rcp.approx.ftz? Plain rcp.approx.f32 has max rel error 2^-23ish? Actually rcp.approx.f32 max ulp error ~1. Fine. I'll use rcp.approx.f32.

Hmm wait, tot can't be zero? sum of exp(x - max) >= exp(0) = 1 (the max element gives 1). So >= 1. Good.

Pass 3:

```python
with txl.serial(ntiles) as it:
    base = (it * BLOCK + tid) * 4
    with txl.if_(base + 4 <= n), txl.then_():
        txl.ptx["ld.shared.v4.f32"](xf..., xs.ptr_to([base]))
        for k: xf[k] = xf[k] * inv — via txl.assign? xf is alloc_local array; example did txl.assign(acc[0], acc[0] + ...) so txl.assign(xf[k], xf[k] * inv) works.
        txl.ptx["st.global.v4.f32"](y.ptr_to([r, base]), xf...)
for k in range(rem):
    idx...
    with txl.if_(tid == k):
        ld.shared.f32(xf[0], xs.ptr_to([idx]))
        txl.assign(xf[0], xf[0] * inv)
        txl.ptx["st.global.f32"](y.ptr_to([r, idx]), xf[0])
```

Note: pass3 reads smem written in pass2 by possibly different threads? No — same thread wrote same indices in pass 2 (same mapping). Actually same thread reads what it wrote, so no barrier strictly needed, but block-sum reduction bar.syncs anyway. Fine.

Could store inv multiplication via mul.f32 DPS: txl.ptx["mul.f32"](xf[k], xf[k], inv). Use that for consistency.

Non-smem path (n > 12288): three global passes.

```python
# pass 1: max over global
if ntiles > 0:
    with txl.serial(ntiles) as it:
        base = (it * BLOCK + tid) * 4
        with txl.if_(base + 4 <= n):
            ld.global.v4.f32(xf...)
            max updates
for k in range(rem): ... global scalar loads with tid==k guard
mx = block_reduce(m, red, fmax, NEG_INF)

# pass 2: exp + sum, write exp to y
s = 0
loop: ld.global.v4; ex2; add; st.global.v4 to y
tail...
tot = block_reduce(s, red, fadd, 0.0)
inv = rcp

# pass 3: rescale y in place
loop: ld.global.v4 from y; mul; st.global.v4 to y
tail...
```

Note pass 3 read/write y same addresses by same thread — no barrier needed.

Actually can we skip separate pass3 by fusing? Not without knowing sum beforehand. Fine.

Alternatively, for the large-n path, note each thread reads its own elements only, so we could keep exp values in... no, too many. Fine.

Now, one more consideration: `x.ptr_to([r, base])` — r is cta_id (int32?), base int32. Static shape so index arithmetic in 32-bit, good.

Potential concern: `txl.serial(ntiles)` with ntiles Python int — fine.

`with txl.if_(base + 4 <= n)` — base is an expression; `base + 4 <= n` where n Python int → expression comparison. Should work.

Comparison `tid == k`: tid is expression, k Python int. Fine.

ex2 operand: `(xf[k] - mx) * log2e` — xf[k] expression (reading local), mx local_scalar read. Arithmetic fine.

Now NEG_INF: define module-level:

```python
NEG_INF_F32 = ... 
```

But it must be constructed inside trace? txl.reinterpret(txl.uint32(0xFF800000), ty="float32") — these constructors can be called at module import? Probably yes since they're pure expression constructors, but tracing context may be needed. Safer to construct inside the kernel body (during trace). txl.float32(...) at module level — the example RMSNorm has eps at module level but only as Python float used inside. I'll construct inside the kernel.

Also careful: `txl.uint32(0xFF800000)` — 0xFF800000 = 4286578688 fits in uint32. Fine.

Now warp reduction on local_scalar `val`: `_warp_reduce` modifies val in place. For block_reduce's second stage I create v = local_scalar and reuse.

One subtlety: in `_block_reduce`, `lane` and `warp` — use txl.lane_id() and txl.warp_id()? Guide says txl.lane_id, txl.warp_id are entry-owned coordinates. Use those: lane = txl.lane_id(), warp = txl.warp_id(). Are they int32 scalars? Yes presumably. Comparisons fine.

Also block reduce when BLOCK=128 → 4 warps; red_smem size 4; the "lane < 4" constant 4 = BLOCK//32. I'll parametrize: NW = BLOCK // 32.

Edge: what if n < 128 etc. — all handled by guards. n=1: ntiles=0 (skip vector loop), rem=1, thread 0 loads x[r,0]. m=-inf then max → x value. Good. But serial(0) — guard with `if ntiles > 0:` in Python. Also for rem loop `for k in range(rem)` — rem=0 → nothing.

Also `txl.serial` requires positive extent? Guard anyway.

Now double-check DPS form for ptx with string name: `txl.ptx["max.f32"](dst, a, b)` — example: `txl.ptx[LD_G_V4](iw[0], ..., ptr)` and `txl.ptx[MUL_F32X2](pk, a, b)`. Good.

ex2.approx.f32: `txl.ptx["ex2.approx.f32"](xf[k], src)` — DPS with dest first. mov.b32 example: `txl.ptx.mov.b32(dst, src)`. Good.

rcp.approx.f32: `txl.ptx["rcp.approx.f32"](inv, tot)`.

Hmm, actually should I worry about ftz? ex2.approx.f32 flushes denormals? Inputs after subtracting max are ≤ 0; exp of very negative → 0 (underflow to zero, fine). Denormal outputs flushed — negligible. OK.

Numerical stability: x ~ 100, subtract max first. Good.

Now smem threshold: A100 max shared per block 163KB (opt-in), default 48KB static. txl smem_pool — unknown whether it emits dynamic smem with opt-in. Keep threshold at 48KB: n <= 12288. Also plus red[4] (16 bytes). Fine.

But also consider: for performance with small n (e.g., n=128 or less), 128 threads with vec4: only 32 threads active in pass. Fine.

Alternative performance improvement: use `ld.global.nc.v4.f32` (non-coherent / __ldg) for read-only x. Could add `.nc`: "ld.global.nc.v4.f32". That's valid PTX. I'll use nc for x loads (read-only). And st.global for y. Use "ld.global.nc.f32" for scalar tail too. It's safe since x isn't written by kernel. I'll include it.

Also could add `@__ldg__`... fine.

Vectorized 16-byte alignment: ptr_to([r, base]) with base multiple of 4 → byte offset multiple of 16 relative to row start; row start offset = r*n*4 bytes — if n not multiple of 4, rows aren't 16B aligned! But vec path only used when base+4<=n... wait the vector loop uses vec4 whenever there are ≥4 consecutive elements at base — that's any n≥4, not just n%4==0. If n%4 != 0, row r starts at byte offset r*n*4 which is not multiple of 16, so ld.global.v4.f32 would be misaligned → error!

Hmm. base multiple of 4 means within-row offset base*4 is 16B-aligned, but the absolute address is x_base + (r*n + base)*4. Alignment requires (r*n + base) % 4 == 0 for all r → need n % 4 == 0.

So vectorization with vec4 requires n % 4 == 0. Options:
- If n % 4 == 0: use vec4 path (all elements, rem=0).
- Else: use scalar path (vec=1): each thread handles strided elements idx = it*128 + tid. Or use vec2 if n%2==0 (8B alignment: need n%2==0). 

So choose vec = 4 if n%4==0, 2 if n%2==0, else 1. Alignment for vec2: (r*n+base) even for all r iff n even. Good.

Actually more general: we could vectorize the "full vector" portion within each row independently... the row base is r*n; within row, elements [0, nv*vec) aligned relative to row start but absolute alignment needs row start aligned to vec. Since row start offset r*n*4, need n*4*r % (vec*4) == 0 ∀r → n % vec == 0. So yes, vec = largest of {4,2,1} dividing n.

Alternatively, treat x as flat and use vector loads across row boundaries? No—each CTA handles one row; crossing rows would break correctness. Keep vec by divisibility.

So:

```python
if n % 4 == 0:
    vec = 4
elif n % 2 == 0:
    vec = 2
else:
    vec = 1
```

For vec=2: ld.global.v2.f32, st.global.v2.f32, ld.shared.v2.f32, st.shared.v2.f32. Does PTX have v2 for f32 shared? ld.shared.v2.f32 exists. Good.

Generalize code with vec as Python constant:

nvec_groups = n // vec (exact since vec | n), ntiles = ceil(nvec_groups / 128), rem = 0 always! Since vec divides n. 

So the tail handling disappears. rem = n - ntiles*... no: nvec_groups*vec == n exactly. Every element covered by vector loop with guard base < n (base = (it*128+tid)*vec; guard base + vec <= n ⟺ base < n since multiples). Guard: base < n.

This simplifies a lot. n odd → vec=1, scalar loads — slower but correct. Could handle odd n with vec4 on the first n-? ... misaligned issue again unless we align per-row: e.g., for odd n, process elements [0, n-?) with scalar until 16B-aligned point? The alignment point differs per row (r*n mod 4 varies). Too complex; scalar fallback is acceptable ("n often not a multiple of 128" — divisibility by 4 typical). Actually we could do: peel leading scalars until aligned: each thread... complexity high. Keep simple: vec ∈ {4,2,1}.

Hmm, but what about performance for odd n? Scalar loads/stores, ~4x traffic instructions. Acceptable.

Let me also double check `txl.ptx["ld.global.v2.f32"](a, b, ptr)` DPS shape — vector load with 2 dests. Presumably generic. And "st.shared.v2.f32"(ptr, a, b).

Alternatively, avoid v2 and just use vec ∈ {4, 1}: n%4==0 → vec4, else scalar. n%2==0 but not 4 → scalar. Simpler, less risk of unsupported v2 forms? v2.f32 is standard PTX. I'll include v2 for better coverage... risk: txl ptx instruction parsing may expect specific signatures but it seems generic (it passed v4.b32 with 4 dests). I'll include vec=2 path.

Hmm, actually to reduce risk let me reconsider: is generic string-indexed ptx fully generic? The example shows txl.ptx[LD_G_V4] with LD_G_V4="ld.global.v4.b32" and 4 dests + ptr, and MUL_F32X2 with (pk, a, b). Seems the ptx namespace builds instructions from name + args generically. I'll trust it for v2.

Now write the unified code parameterized by vec and use_smem.

Instruction names depending on vec:
- G_LD = f"ld.global.nc.v{vec}.f32"
- G_LD_VEC... For scalar (vec=1): "ld.global.nc.f32" (no v1). PTX: ld.global.nc.f32 is valid. For vec=1 use plain.

Define:

```python
if vec == 1:
    LD_G = "ld.global.nc.f32"
    ST_G = "st.global.f32"
    LD_S = "ld.shared.f32"
    ST_S = "st.shared.f32"
else:
    LD_G = f"ld.global.nc.v{vec}.f32"
    ST_G = f"st.global.v{vec}.f32"
    LD_S = f"ld.shared.v{vec}.f32"
    ST_S = f"st.shared.v{vec}.f32"
```

And calls:

```python
def ld_g(ptr):
    if vec == 1:
        txl.ptx[LD_G](xf[0], ptr)
    elif vec == 2:
        txl.ptx[LD_G](xf[0], xf[1], ptr)
    else:
        txl.ptx[LD_G](xf[0], xf[1], xf[2], xf[3], ptr)
```

Python-level helper closures inside make_kernel (before @txl.kernel or inside). They run at trace time. Define inside kernel body for access to xf etc. Actually define at make_kernel level taking xf and ptr.

Let me now write the whole module.

```python
import tirx_kernels.tirx_lite as txl

BLOCK = 128
WARPS = BLOCK // 32
FULL_MASK = 0xFFFFFFFF
LOG2E = 1.4426950408889634
NEG_INF_BITS = 0xFF800000
SMEM_LIMIT_ELEMS = 12288  # 48 KiB of float32 staging


def _cta_sync():
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))


def _fmax(dst, a, b):
    txl.ptx["max.f32"](dst, a, b)


def _fadd(dst, a, b):
    txl.ptx["add.f32"](dst, a, b)


def _warp_reduce(val, combine):
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer,
            txl.reinterpret(val, ty="uint32"),
            txl.uint32(delta),
            txl.uint32(31),
            txl.uint32(FULL_MASK),
        )
        combine(val, val, txl.reinterpret(peer, ty="float32"))


def _block_reduce(val, red, combine, identity):
    lane = txl.lane_id()
    warp = txl.warp_id()
    _warp_reduce(val, combine)
    with txl.if_(lane == 0), txl.then_():
        txl.ptx["st.shared.f32"](red.ptr_to([warp]), val)
    _cta_sync()
    with txl.if_(warp == 0), txl.then_():
        part = txl.local_scalar("float32")
        with txl.if_(lane < WARPS), txl.then_():
            txl.ptx["ld.shared.f32"](part, red.ptr_to([lane]))
        with txl.else_():
            txl.assign(part, identity)
        _warp_reduce(part, combine)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx["st.shared.f32"](red.ptr_to([0]), part)
    _cta_sync()
    out = txl.local_scalar("float32")
    txl.ptx["ld.shared.f32"](out, red.ptr_to([0]))
    return out
```

Wait: `txl.assign(part, identity)` where identity is an expression (NEG_INF reinterpret). Should work — example: txl.assign(ss[0], txl.float32(0.0)).

Note: `_warp_reduce(part, ...)` — part is local_scalar("float32"); combine(part, part, ...) → txl.ptx["max.f32"](part, part, x) — local_scalar as ptx dest: example used peer (local_scalar) as shfl dest and `rms` local_scalar as rsqrt dest? Example: `txl.assign(rms, txl.rsqrt(...))` — assign, not ptx dest. But shfl dest peer is local_scalar, so ptx DPS accepts local_scalar dest. OK.

Also careful: `_block_reduce` uses `red.ptr_to([warp])` — warp is expression. Fine.

Now make_kernel:

```python
def make_kernel(rows: int, n: int):
    if n % 4 == 0:
        vec = 4
    elif n % 2 == 0:
        vec = 2
    else:
        vec = 1
    groups = n // vec          # exact
    ntiles = (groups + BLOCK - 1) // BLOCK
    use_smem = n <= SMEM_LIMIT_ELEMS

    if vec == 1:
        LD_G = "ld.global.nc.f32"
        ST_G = "st.global.f32"
        LD_S = "ld.shared.f32"
        ST_S = "st.shared.f32"
    else:
        LD_G = f"ld.global.nc.v{vec}.f32"
        ST_G = f"st.global.v{vec}.f32"
        LD_S = f"ld.shared.v{vec}.f32"
        ST_S = f"st.shared.v{vec}.f32"

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()

        smem = txl.smem_pool()
        red = smem.alloc([WARPS], "float32")
        xs = smem.alloc([n], "float32") if use_smem else None

        xf = txl.alloc_local([vec], "float32")
        m = txl.local_scalar("float32")
        neg_inf = txl.reinterpret(txl.uint32(NEG_INF_BITS), ty="float32")
        log2e = txl.float32(LOG2E)

        def ld_global(ptr):
            if vec == 1:
                txl.ptx[LD_G](xf[0], ptr)
            elif vec == 2:
                txl.ptx[LD_G](xf[0], xf[1], ptr)
            else:
                txl.ptx[LD_G](xf[0], xf[1], xf[2], xf[3], ptr)

        def st_global(ptr):
            if vec == 1:
                txl.ptx[ST_G](ptr, xf[0])
            elif vec == 2:
                txl.ptx[ST_G](ptr, xf[0], xf[1])
            else:
                txl.ptx[ST_G](ptr, xf[0], xf[1], xf[2], xf[3])

        def ld_shared(ptr):
            ... similarly with LD_S

        def st_shared(ptr):
            ... ST_S

        # ---- pass 1: max (+ stage into smem when it fits) ----
        txl.assign(m, neg_inf)
        if ntiles > 0:
            with txl.serial(ntiles) as it:
                base = (it * BLOCK + tid) * vec
                with txl.if_(base < n), txl.then_():
                    ld_global(x.ptr_to([r, base]))
                    for k in range(vec):
                        txl.ptx["max.f32"](m, m, xf[k])
                    if use_smem:
                        st_shared(xs.ptr_to([base]))

        mx = _block_reduce(m, red, _fmax, neg_inf)

        # ---- pass 2: exp(x - mx), sum ----
        s = txl.local_scalar("float32")
        txl.assign(s, txl.float32(0.0))
        if ntiles > 0:
            with txl.serial(ntiles) as it:
                base = (it * BLOCK + tid) * vec
                with txl.if_(base < n), txl.then_():
                    if use_smem:
                        ld_shared(xs.ptr_to([base]))
                    else:
                        ld_global(x.ptr_to([r, base]))
                    for k in range(vec):
                        txl.ptx["ex2.approx.f32"](xf[k], (xf[k] - mx) * log2e)
                        txl.ptx["add.f32"](s, s, xf[k])
                    if use_smem:
                        st_shared(xs.ptr_to([base]))
                    else:
                        st_global(y.ptr_to([r, base]))

        tot = _block_reduce(s, red, _fadd, txl.float32(0.0))
        inv = txl.local_scalar("float32")
        txl.ptx["rcp.approx.f32"](inv, tot)

        # ---- pass 3: scale and store ----
        if ntiles > 0:
            with txl.serial(ntiles) as it:
                base = (it * BLOCK + tid) * vec
                with txl.if_(base < n), txl.then_():
                    if use_smem:
                        ld_shared(xs.ptr_to([base]))
                    else:
                        ld_global(y.ptr_to([r, base]))  # non-smem: y holds exp values
                    for k in range(vec):
                        txl.ptx["mul.f32"](xf[k], xf[k], inv)
                    st_global(y.ptr_to([r, base]))

    return softmax_txl
```

Wait — in non-smem pass 3, ld_global reads from y with ld.global.nc — y is being written by this kernel; `.nc` (non-coherent, read-only texture path) on data written by same kernel is NOT safe (ldg requires data not written during kernel). So for pass 3 non-smem, must use plain ld.global (coherent). Define separate loader or parametrize LD_G with a flag. I'll make ld_global take a name parameter:

```python
def ld_vec(name, ptr):
    if vec == 1:
        txl.ptx[name](xf[0], ptr)
    ...
```

And use LD_G_NC = "ld.global.nc..." for x, LD_G = "ld.global..." for y. In non-smem pass2 we write exp to y and pass3 reads y written by same thread — plain ld.global works (same thread, program order). Good.

Also in non-smem path, pass 3 loads y written by the same thread in pass 2 — but between them, block_reduce barriers ran; same-thread writes visible to itself regardless. Fine.

Hmm — one more concern for the non-smem path: pass 3 read-then-write same location; fine.

Also for correctness in non-smem pass2: threads write y then pass3 reads/writes. Since same thread, no cross-thread hazard.

But wait: bar.sync and memory visibility for global? Same thread only — no issue.

Now check: `txl.ptx["ex2.approx.f32"](xf[k], (xf[k] - mx) * log2e)` — dest xf[k] (alloc_local element) as DPS dest: example `txl.ptx.mov.b32(dst[2*i], ...)` uses alloc_local element as dest. Good. Source expression arithmetic — example passes expressions? `txl.ptx[CVT_F16X2](ow[pair], xf[2*pair+1], xf[2*pair])` — sources are elements, not compound expressions. Will ptx source accept compound expression `(xf[k] - mx) * log2e`? Probably it builds operand values; expressions should be materialized into temporaries automatically. Hmm, risk. To be safe, materialize into a temp local:

```python
t = xf[k] - mx
txl.ptx["mul.f32"](t, t, log2e)
txl.ptx["ex2.approx.f32"](xf[k], t)
```

where t = txl.local_scalar("float32") declared once outside loops. Using local_scalar `t` as dest then source. Also `txl.assign(t, xf[k] - mx)` to be extra safe with assign (definitely supports expressions). Then mul.f32(t, t, log2e), ex2(xf[k], t). I'll do that.

Similarly `txl.assign(m, neg_inf)` where neg_inf is a reinterpret expression — assign should handle expression values (example assigns ss[0] from ld dest etc., and txl.assign(rms, txl.rsqrt(...)) expression). Fine.

`txl.assign(part, identity)` in _block_reduce — identity is expression (neg_inf reinterpret or float32(0.0)). Fine.

Now: does `txl.ptx["max.f32"]` exist as valid spelled instruction? PTX ISA: max.f32 d, a, b — yes (with optional ftz). Fine on sm_80.

`ex2.approx.f32 d, a` — yes.
`rcp.approx.f32 d, a` — yes.
`bar.sync a{, b}` — yes.
`shfl.sync.bfly.b32 d, a, b, c, membermask` — example confirms.
`ld.global.nc.v4.f32` — yes.
`st.global.v4.f32` — yes.
`ld.shared.v4.f32`, `st.shared.v4.f32` — example confirms.
`ld.shared.v2.f32`, `st.shared.v2.f32` — standard.
`ld.global.v2.f32`, `st.global.v2.f32` — standard.

Potential issue: txl.ptx attribute style for shfl: example used `txl.ptx.shfl_sync.bfly.b32`. I'll keep that exact spelling.

bar.sync: `txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))` — matches example cta_sync(bar_id) with two args.

Now `txl.lane_id()` and `txl.warp_id()` — guide says these exist. In the example, lane = tid & 31, warp = tid >> 5. Either fine. I'll use lane_id()/warp_id()... hmm, warp_id returns warp id within block. Yes. But is the comparison `lane < WARPS` with lane expression and Python int fine? Yes.

Actually wait — in `_block_reduce`, I read `red.ptr_to([lane])` where lane < WARPS guard. red has WARPS elements. Good.

One more concern: `_block_reduce` is called twice; each call declares new locals — fine. But `neg_inf` identity: each call passes expression — fine.

Edge case n where ntiles == 0: impossible since n>=1 → groups>=1 → ntiles>=1. Actually groups >= 1 always (n>=1), so ntiles >= 1. The `if ntiles > 0` guard unnecessary but harmless. Keep it simple: always emit serial (ntiles>=1). Actually if vec chosen s.t. groups = n//vec >= 1. Yes. I'll drop the guard... keep it anyway harmless? serial(0) never occurs. I'll just emit without guard for cleanliness. Hmm, but if someone calls n=0? Task says n>=1. Fine.

Now, one subtle thing: `with txl.if_(base < n)` where base is `(it * BLOCK + tid) * vec`. it is loop var expression. Fine.

Type of `r = txl.cta_id()`: scalar. `x.ptr_to([r, base])` — mixing int32 fine.

Another perf consideration: for the smem path with small n, e.g., n=64: only 16 threads active with vec4. Fine.

Consider whether we should use `min_blocks_per_sm`? Not needed.

Also `backend_config={"cuda": {"arch": "sm_80"}}` — given in interface. Keep.

Double-check the annotation style: `def softmax_txl(x: txl.Tensor((rows, n), "float32"), ...)`. Given in prompt. Keep exactly.

Potential issue: helpers defined at module level (_warp_reduce etc.) called from within traced kernel — guide says helpers using native T namespace work with line tracing; "Statements and instruction calls traced from file-backed Python code carry TIRx source spans". Module-level helpers should be fine (RMSNorm defined helpers inside kernel body, but module-level python helpers that emit IR should also work since tracing is active). The guide: "Emission and source locations also work for helpers using the native T namespace." OK.

But careful: _block_reduce contains `with txl.if_(...)` control flow — emitting control flow from helpers should be fine too.

One risk: `txl.local_scalar("uint32")` inside _warp_reduce called multiple times — fine.

Also within `with txl.if_(warp == 0), txl.then_():` we declare `part = txl.local_scalar(...)` inside the branch — allocations inside control flow: local alloc inside if is fine (registers). OK.

Now the non-smem path details: LD from y in pass3 must be coherent plain "ld.global.f32"/"ld.global.v2.f32"/"ld.global.v4.f32". Define both name sets:

```python
if vec == 1:
    LDG_NC = "ld.global.nc.f32"; LDG = "ld.global.f32"
    STG = "st.global.f32"
    LDS = "ld.shared.f32"; STS = "st.shared.f32"
else:
    LDG_NC = f"ld.global.nc.v{vec}.f32"; LDG = f"ld.global.v{vec}.f32"
    STG = f"st.global.v{vec}.f32"
    LDS = f"ld.shared.v{vec}.f32"; STS = f"st.shared.v{vec}.f32"
```

And loader/storer closures parameterized by instruction name.

Wait, one more correctness point for the smem path pass 3: threads read smem locations they themselves wrote in pass 2 (same mapping), so no barrier needed beyond what reduction provides; anyway reduction's final bar.sync + ld covers.

Hmm, actually pass 2 stores to smem then _block_reduce for sum runs: warp reduce (no smem), lane0 st red[warp], bar.sync → data stores visible. Then pass 3 reads own data. Fine.

Actually wait — potential hazard in pass 2 smem path: thread A stores xs[base] in pass2 tile i; but does any other thread read xs[base] before? Pass 2 reads xs[base] by the same thread that wrote it in pass 1 (same base mapping) — but pass 1 writes and pass 2 reads are separated by _block_reduce's bar.sync. Good.

Now also consider whether `txl.serial` loop variable arithmetic `it * BLOCK` — it is int32; BLOCK Python int. Fine.

Let me also think about whether to use `txl.float32(LOG2E)` — LOG2E Python float, fine.

Another thought: use fma for (x-mx)*log2e? ptx fma.rn.f32? mul+sub then ex2. Current: assign t = xf[k]-mx; mul.f32 t, t, log2e. Two instructions. Could do: txl.ptx["sub.f32"](t, xf[k], mx); txl.ptx["fma.rn.f32"]? Actually (x - mx) * log2e = fma(x, log2e, -mx*log2e) — one instruction: fma.rn.f32(t, xf[k], log2e, neg_log2e_mx) where neg_log2e_mx = -(mx*log2e) computed once per row. That saves an instruction per element. Let's do: after mx computed, compute shift = -(mx * log2e) once (local scalar): txl.ptx["mul.f32"](shift, mx, log2e); txl.ptx["neg.f32"](shift, shift). Then per element: txl.ptx["fma.rn.f32"](t, xf[k], log2e, shift); ex2. Nice. Accuracy: fine.

neg.f32 exists. Or fma with negation: fma.rn.f32 t, xf[k], -log2e... no, shift approach is fine. Alternatively mul by -log2e once: shift = mx * (-LOG2E) via txl.assign(shift, mx * txl.float32(-LOG2E))? Simplest: compute nlog2e = txl.float32(-LOG2E) and mul.f32(shift, mx, nlog2e). Then fma(t, xf[k], log2e, shift) → t = x*log2e + shift = (x - mx)*log2e. 

Precision concern: x*log2e - mx*log2e vs (x-mx)*log2e: tiny rounding differences, irrelevant for softmax.

OK.

Now also pass 2 sum accumulation: add.f32 per element into s. Fine.

Let me now also double check `txl.ptx["fma.rn.f32"](t, a, b, c)` — DPS with 4 operands (dest + 3 srcs). Should be fine generically.

Hmm, is there risk the generic ptx dispatcher requires knowing instruction signatures? The examples show varied arities (ld v4: 5 args; mul f32x2: 3; cvt: 3; mov: 2; shfl: 5; bar.sync: 2). It seems generic: name string + args. Confident enough.

One thing: the example used `txl.ptx.st.global_.f32(...)` attribute style where "global_" presumably because... maybe attribute access maps `global_` → `global`? Not sure of the exact mapping (maybe underscores→dots). String indexing avoids the question. Use strings everywhere except shfl_sync and bar.sync which I copy exactly from the example.

Now, smem threshold: also account that the red array adds 16 bytes. n <= 12288 → 49152 bytes + 16 → slightly over 48KB? 12288*4 = 49152 = 48KB exactly, plus red 16 bytes = 49168 > 49152! If the 48KB limit applies strictly, this fails. Set SMEM_LIMIT_ELEMS = 12280 (49120 bytes + 16 = 49136 < 49152). Let's use 12280. Or simply 12000. I'll use 12280.

Actually, does txl smem_pool emit dynamic shared memory with cudaFuncSetAttribute for >48KB? Unknown. Keep conservative.

Also, for n between, say, 8192 and 12280, using 32-48KB smem reduces occupancy but softmax rows that big are memory-bound anyway; fine.

Now, non-smem path: pass 2 writes exp to y globally, pass 3 rescales. Global traffic: pass1 read x (nc), pass2 read x (nc), write y, pass3 read y, write y. Total 5 accesses/elem vs smem path's 1R+1W global + smem traffic. Fine for huge n.

Edge: rows could be huge; grid=rows fine.

Now write final code carefully.

One more consideration: `txl.assign(m, neg_inf)` — m is local_scalar; example assigns to local_scalar rms via txl.assign. Good.

Also `txl.ptx["max.f32"](m, m, xf[k])` — m local_scalar as dest and src. shfl example used acc[0] alloc_local; peer local_scalar as dest. local_scalar as src: `txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(acc[0]...))` — reinterpret of local as src value. And combine call in example: `txl.assign(acc[0], acc[0] + reinterpret(peer))` — they used assign, not ptx add. Hmm, no ptx example with local_scalar as both dest and src operand. mov.b32(dst_elem, txl.cuda.float2_x(pk)) — dest is local array elem, source expression. I think local_scalar as ptx operand (value) works since reinterpret(val) in shfl source worked with acc[0] (alloc_local elem). For local_scalar as source: `txl.ptx["max.f32"](m, m, xf[k])` — m as source. reinterpret(m) would work surely; plain m as a float operand to a float instruction should also work (it's a value expression). OK.

Alternatively to be safer, I could accumulate max via Python-level expressions and txl.assign: `txl.assign(m, txl_max(m, xf[k]))` — but no known txl.max. Use Select: txl.Select(m >= xf[k], m, xf[k])? Select is documented. Hmm, but Select generates select instruction — fine too. I'll stick with ptx max.f32; it's standard.

Let me also handle: what if `txl.ptx["ld.global.nc.v2.f32"]`... fine.

Write the vec-generic helper closures:

```python
def _ld_vec(name, ptr):
    if vec == 1:
        txl.ptx[name](xf[0], ptr)
    elif vec == 2:
        txl.ptx[name](xf[0], xf[1], ptr)
    else:
        txl.ptx[name](xf[0], xf[1], xf[2], xf[3], ptr)

def _st_vec(name, ptr):
    if vec == 1:
        txl.ptx[name](ptr, xf[0])
    elif vec == 2:
        txl.ptx[name](ptr, xf[0], xf[1])
    else:
        txl.ptx[name](ptr, xf[0], xf[1], xf[2], xf[3])
```

These close over xf and vec — define inside kernel body.

Full module:

```python
import tirx_kernels.tirx_lite as txl

BLOCK = 128
WARPS = BLOCK // 32
FULL_MASK = 0xFFFFFFFF
LOG2E = 1.4426950408889634
NEG_INF_BITS = 0xFF800000
SMEM_LIMIT_ELEMS = 12280  # 48 KiB budget minus the reduction scratch


def _cta_sync():
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))


def _fmax(dst, a, b):
    txl.ptx["max.f32"](dst, a, b)


def _fadd(dst, a, b):
    txl.ptx["add.f32"](dst, a, b)


def _warp_reduce(val, combine):
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer,
            txl.reinterpret(val, ty="uint32"),
            txl.uint32(delta),
            txl.uint32(31),
            txl.uint32(FULL_MASK),
        )
        combine(val, val, txl.reinterpret(peer, ty="float32"))


def _block_reduce(val, red, combine, identity):
    lane = txl.lane_id()
    warp = txl.warp_id()
    _warp_reduce(val, combine)
    with txl.if_(lane == 0), txl.then_():
        txl.ptx["st.shared.f32"](red.ptr_to([warp]), val)
    _cta_sync()
    with txl.if_(warp == 0), txl.then_():
        part = txl.local_scalar("float32")
        with txl.if_(lane < WARPS), txl.then_():
            txl.ptx["ld.shared.f32"](part, red.ptr_to([lane]))
        with txl.else_():
            txl.assign(part, identity)
        _warp_reduce(part, combine)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx["st.shared.f32"](red.ptr_to([0]), part)
    _cta_sync()
    out = txl.local_scalar("float32")
    txl.ptx["ld.shared.f32"](out, red.ptr_to([0]))
    return out


def make_kernel(rows: int, n: int):
    if n % 4 == 0:
        vec = 4
    elif n % 2 == 0:
        vec = 2
    else:
        vec = 1
    groups = n // vec
    ntiles = (groups + BLOCK - 1) // BLOCK
    use_smem = n <= SMEM_LIMIT_ELEMS

    if vec == 1:
        LDG_RO = "ld.global.nc.f32"
        LDG = "ld.global.f32"
        STG = "st.global.f32"
        LDS = "ld.shared.f32"
        STS = "st.shared.f32"
    else:
        LDG_RO = "ld.global.nc.v%d.f32" % vec
        LDG = "ld.global.v%d.f32" % vec
        STG = "st.global.v%d.f32" % vec
        LDS = "ld.shared.v%d.f32" % vec
        STS = "st.shared.v%d.f32" % vec

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()

        smem = txl.smem_pool()
        red = smem.alloc([WARPS], "float32")
        xs = smem.alloc([n], "float32") if use_smem else None

        xf = txl.alloc_local([vec], "float32")
        t = txl.local_scalar("float32")
        m = txl.local_scalar("float32")
        neg_inf = txl.reinterpret(txl.uint32(NEG_INF_BITS), ty="float32")
        log2e = txl.float32(LOG2E)
        nlog2e = txl.float32(-LOG2E)

        def ld_vec(name, ptr):
            if vec == 1:
                txl.ptx[name](xf[0], ptr)
            elif vec == 2:
                txl.ptx[name](xf[0], xf[1], ptr)
            else:
                txl.ptx[name](xf[0], xf[1], xf[2], xf[3], ptr)

        def st_vec(name, ptr):
            if vec == 1:
                txl.ptx[name](ptr, xf[0])
            elif vec == 2:
                txl.ptx[name](ptr, xf[0], xf[1])
            else:
                txl.ptx[name](ptr, xf[0], xf[1], xf[2], xf[3])

        # pass 1
        txl.assign(m, neg_inf)
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                ld_vec(LDG_RO, x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["max.f32"](m, m, xf[k])
                if use_smem:
                    st_vec(STS, xs.ptr_to([base]))

        mx = _block_reduce(m, red, _fmax, neg_inf)

        # shift so that ex2(x * log2e + shift) == exp(x - mx)
        shift = txl.local_scalar("float32")
        txl.ptx["mul.f32"](shift, mx, nlog2e)

        # pass 2
        s = txl.local_scalar("float32")
        txl.assign(s, txl.float32(0.0))
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                if use_smem:
                    ld_vec(LDS, xs.ptr_to([base]))
                else:
                    ld_vec(LDG_RO, x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["fma.rn.f32"](t, xf[k], log2e, shift)
                    txl.ptx["ex2.approx.f32"](xf[k], t)
                    txl.ptx["add.f32"](s, s, xf[k])
                if use_smem:
                    st_vec(STS, xs.ptr_to([base]))
                else:
                    st_vec(STG, y.ptr_to([r, base]))

        tot = _block_reduce(s, red, _fadd, txl.float32(0.0))
        inv = txl.local_scalar("float32")
        txl.ptx["rcp.approx.f32"](inv, tot)

        # pass 3
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                if use_smem:
                    ld_vec(LDS, xs.ptr_to([base]))
                else:
                    ld_vec(LDG, y.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["mul.f32"](xf[k], xf[k], inv)
                st_vec(STG, y.ptr_to([r, base]))

    return softmax_txl
```

Wait: fma computes t = xf[k]*log2e + shift where shift = mx * (-log2e). t = (x - mx)*log2e. Correct.

Issue: `xs = smem.alloc([n], "float32") if use_smem else None` — the conditional expression at trace time: if use_smem is Python bool constant, fine (n is static int). Good.

Issue: `txl.serial(ntiles)` — ntiles Python int >= 1. Good.

Issue: pass1 in smem path — after loop, _block_reduce for max: it uses bar.sync which also makes pass-1 smem stores visible. Good.

But wait, there's a subtle correctness issue in the non-smem path pass 2 → pass 3: pass 2 writes y via st.global (weak). pass 3 reads y via ld.global (weak, same thread). Same thread read-after-write to same address in PTX: guaranteed? Within a thread, memory operations to the same address are ordered per PTX's program-order semantics for same-thread same-address (yes, "same address accesses by same thread are ordered"). Fine.

Also non-smem pass 3 uses ld.global (coherent) not .nc — correct since data written this kernel.

One more: ex2.approx.f32 argument range: (x-mx) ∈ (-inf, 0], *log2e fine. If x contains -inf? x ~ 100 magnitude, fine.

NaN handling: not required.

Check n=1: vec=1, groups=1, ntiles=1. Thread 0: base=0 < 1 → load x[r,0]; others base>=1 skip. m over threads: thread0 has value, others -inf → mx correct. Pass2: thread0 e=1.0, s=1. Others 0. tot=1. inv=1. Pass3: thread0 stores 1.0. Correct.

Check n=6: vec=2, groups=3, ntiles=1. bases: tid*2 for tid 0..2 → 0,2,4 <6; tid>=3 skip. Correct.

Check n=130: vec=2? 130%4=2 → vec=2, groups=65, ntiles=1: bases 0..128 step2 → tid 0..64 cover 0..128; tid 65 base 130 not <130. Correct. Hmm wait tid 64 → base 128 <130, covers 128,129. Good.

Check alignment vec=2: n even → row offset r*n even elements → 8B aligned. ld.global.nc.v2.f32 needs 8B alignment. Good. vec=4: n%4==0 → 16B aligned. Good.

Check smem path n=12280: 12280*4 = 49120 + 16 = 49136 ≤ 49152. OK. n=12281 (odd): vec=1, use_smem False → non-smem path. n=12282: vec=2, 49128+16 = 49144 ≤ 49152 ok. n=12284: vec=4, 49136+16=49152 exactly = 48KB. Exactly at limit — is 49152 allowed as static limit? The 48KB is the max static shared per block; dynamic requires opt-in beyond 48KB. 49152 = 48*1024 exactly, allowed. OK but smem_pool might be dynamic shared memory... if dynamic, default limit is also 48KB without attribute. Exactly 48KB dynamic is allowed only with cudaFuncSetAttribute? Actually dynamic smem >48KB requires opt-in; ==48KB is fine for dynamic too? The default maximum dynamic shared memory is 48KB (cudaDevAttrMaxSharedMemoryPerBlockOptin is 163KB on A100; without opt-in, 48KB applies to both static and dynamic). Exactly 49152 dynamic is allowed without opt-in. OK, safe. But to have margin, set SMEM_LIMIT_ELEMS = 12284 → max 49152. Fine, keep 12284? vec divides n so worst case n=12284 vec=4: 49136+16 = 49152. Fine. I'll set limit 12284.

Hmm, actually also note: smem_pool might handle opt-in automatically (many DSLs do). Whatever — conservative threshold works.

Another perf thought: for the smem path, is staging in smem actually beneficial vs re-reading global? For rows fitting in L2 (n ≤ 12280 → 49KB per row; multiple rows — total matrix may exceed L2 40MB... with grid=rows CTAs all resident simultaneously reading; second read likely L2-hit if row is re-read soon — but CTAs process concurrently, x row read in pass2 shortly after pass1, likely still in L2). smem path avoids the uncertainty and is standard (e.g., NVIDIA's softmax uses registers/smem). Keep smem.

Occupancy: 48KB smem → 3 CTAs/SM (164KB). Fine.

Now double-check the block reduction correctness when some warps have all-inactive... all 128 threads execute reduction regardless of guards (guards only wrap loads). All threads call _block_reduce — divergent-free. Good.

But wait: in pass 1, the `with txl.if_(base < n)` guard — threads where guard false skip loads but still reach _block_reduce. Control flow reconverges. Good.

One concern: `txl.warp_id()` — returns warp index within CTA for 1D block. Yes.

Another: `txl.lane_id()` returns lane. Yes.

Check `_warp_reduce` on `val` when val is `m` (local_scalar) — reinterpret(val, ty="uint32") fine.

Potential issue: `txl.ptx["st.shared.f32"](red.ptr_to([warp]), val)` — dest ptr from smem tensor ptr_to with runtime index warp. Example: `txl.ptx[ST_S_F32](sum_sq_smem.ptr_to([warp]), ss[0])`. Same pattern. Good.

`txl.ptx["ld.shared.f32"](part, red.ptr_to([lane]))` — same as example. Good.

Now, `txl.ptx["rcp.approx.f32"](inv, tot)` — tot is local returned from _block_reduce. Good.

Also, `mx` is a local_scalar returned by _block_reduce. Using it as operand in mul.f32 — fine.

The `identity` for max is neg_inf expression; but note `_block_reduce(m, red, _fmax, neg_inf)` — the identity only used in the else branch for lanes >= WARPS (warp 0, lanes 4..31). Good.

One more consideration: could two consecutive _block_reduce calls with the same red smem conflict? First call's final read `out = ld.shared red[0]` happens after final bar.sync by all threads — wait: sequence: bar.sync (second), then all threads ld red[0] into out, return. Then second _block_reduce call: warp reduce, lane0 st red[warp]... Thread ordering: the ld red[0] by thread A and subsequent st red[warp] by thread B — is there a race? After second bar.sync, all threads load red[0]. Then they proceed to the next _block_reduce's first st.shared red[warp] (by lane 0s). Could thread B (lane 0 of warp 1) execute st red[1] before thread A (some lane in warp 3) executes its ld red[0]? Yes! Different addresses (red[1] vs red[0]) — no conflict. The only same-address hazard: st red[0] by lane0-warp0 in second reduce vs ld red[0] from first reduce by other threads. Between them there's the first cta_sync (bar.sync) of the second _block_reduce call — wait, order in second call: warp_reduce (shfl only), lane==0: st red[warp] — this st red[0] (warp 0's lane 0) can happen before other threads have done their ld red[0] from the first call? The ld red[0] from first call happens right after the first call's final bar.sync, before returning and before second call's warp reduce. But threads run independently: thread in warp 3 could still be at the ld red[0] while warp0's lane 0 races ahead through shfl and does st red[0]. RACE on red[0]!

In the first _block_reduce: final bar.sync, then out = ld red[0]. In second call: ... st red[0] by warp0-lane0. A slow thread in warp 3 might not yet have executed ld red[0]. Yes, this is a genuine race.

Fix options:
1. Have _block_reduce end with bar.sync after the final read. I.e., add _cta_sync() at the end. Then the second call's stores happen after all threads completed the load. Cost: one extra bar.sync per reduction — negligible.

But wait: with trailing bar.sync, first call: bar.sync; ld red[0]; bar.sync. Second call: st red[warp]... safe.

Also within a single reduction: st red[0] (warp0 lane0, second stage) then bar.sync then ld red[0] — safe.

Also check: does the pass-2/pass-3 smem data hazard rely on reduction barriers? Pass 2 stores to xs, then _block_reduce for sum: its trailing bar.sync also ensures xs stores visible... but same-thread reads anyway. Fine.

Add trailing _cta_sync() in _block_reduce. But then the last call (sum reduction) ends with an unnecessary bar.sync before pass 3 — harmless.

Actually careful about deadlock: bar.sync with all 128 threads — all threads execute _block_reduce uniformly, so barriers are uniform. Good.

2. Alternative: return via reading red[0] but use different smem slots per reduction. Trailing sync is simpler.

Now also reconsider: the first bar.sync inside _block_reduce is before the "warp==0" branch; all fine.

Let me also double check pass1→pass2 for non-smem path: no smem data; fine.

Also potential issue with `txl.if_` conditions using `lane < WARPS` — WARPS=4 Python int. fine.

Now, the trailing sync addition:

```python
    _cta_sync()
    out = txl.local_scalar("float32")
    txl.ptx["ld.shared.f32"](out, red.ptr_to([0]))
    _cta_sync()
    return out
```

Hmm wait, but the ld red[0] then bar.sync: the second bar.sync ensures all threads did the ld before anyone proceeds. Good.

Actually alternative: broadcast via shfl from warp... no, keep it.

Now let me reconsider `txl.ptx["max.f32"]`: Hmm, one more check — is "max.f32" valid without rounding modifier? PTX: max.f32 d, a, Bro; also max.ftz.f32, max.NaN.f32. Plain max.f32 is valid. Yes.

"add.f32" valid. "mul.f32" valid. "fma.rn.f32" valid. "neg.f32" not used now. "ex2.approx.f32" valid. "rcp.approx.f32" valid.

Now consider: should I use ftz variants? Plain is fine.

Also `txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))` — BLOCK=128. Matches example pattern (they passed nthreads). Good.

Now about `txl.alloc_local([vec], "float32")` with vec=1: array of 1. Indexing xf[0] fine.

Also, "statements after the block run on the host" — we use flat device_entry (no with), so entire body is device. Good.

Type annotations: `x: txl.Tensor((rows, n), "float32")` — rows, n are Python ints from closure. The example `make_kernel` in prompt uses exactly this. Good.

Potential issue: the kernel body references Python constants (n, vec, ntiles, names) — traced as constants. Good.

Also `txl.device_entry(launch=..., backend_config=...)` — flat call, matches prompt.

One more: for use_smem path we allocate xs of size n even if n small — fine.

Let me reconsider performance detail: in pass 1 (smem path), we do global load + smem store; pass 2: smem load + compute + smem store; pass 3: smem load + global store. Total smem traffic 3 accesses/elem. Alternative: keep values in registers (per-thread elements) when ntiles small... skip; smem is fine.

Alternatively fuse pass 2 and 3 differently: pass 2 computes exp and sum; pass 3 scales. Standard.

Another micro-opt: `#pragma unroll` handled by compiler.

I think the code is solid. One more check of `txl.reinterpret(txl.uint32(NEG_INF_BITS), ty="float32")` — guide: txl.reinterpret(value, ty=txl.uint32). ty as string used in example: txl.reinterpret(acc[0], ty="uint32"). Good.

`txl.uint32(0xFF800000)` — value 4286578688 < 2^32. OK.

Also: does `txl.float32(-LOG2E)` accept negative float? Should.

Edge: what about `base < n` where base = (it*BLOCK + tid)*vec — types int32. n Python int → constant. fine.

One question: `x.ptr_to([r, base])` — r from cta_id, base expression. Example used out.ptr_to([r, col]) with col = it*128 + thread_id. Same. Good.

Also grid=rows with rows up to 2^31-1 — cta_id int32 fine.

Final check on `_block_reduce` being called while inside... it's called at kernel top-level scope (not inside serial loops) — both calls. Yes, pass 1 loop ends before reduction. Good.

Wait, actually there's an issue: `_warp_reduce` uses shfl_sync with FULL_MASK — requires all 32 lanes of each warp converged. In `_block_reduce` second stage, it's inside `with txl.if_(warp == 0)` — entire warp 0 enters, so warp 0 is converged with full mask. Other warps skip entirely (no shfl). Fine. First stage: all threads converged. Fine.

Also in pass loops, guards `if base < n` create divergence but no shfl inside them. Good.

Now, one thing about `txl.serial(ntiles)` — variable named `it` shadows; fine.

Also need to double check: can Python `for k in range(vec)` unrolled loop bodies emit inside `with txl.if_` then_ block — yes, ordinary Python.

Let me also think: is `txl.ptx["ld.global.nc.f32"]` okay for vec=1? PTX: ld.global.nc.f32 d, [a]. Yes.

Write final answer. Also add brief comments.

One more consideration on the interface: "one kernel is traced per (rows, n)" — our make_kernel traces at decoration time inside make_kernel. Good.

Also should the module avoid other imports — only txl. Yes.

Double-check the reduction second stage: `part` local declared inside if branch — alloc inside conditional region. The RMSNorm example declared locals at top. Declaring a local inside an if might be disallowed? IR-wise, alloc inside if is fine in SSA/TIR. But to be safe, hoist `part` declaration to _block_reduce top (outside branches). Locals are just registers; declaring at function top-level of the helper is fine. Similarly `out`. And `peer` in _warp_reduce is at helper top. Let me hoist part/out to top of _block_reduce.

But _block_reduce is called twice — each call emits fresh locals at kernel scope. Fine.

Also, note `txl.local_scalar` inside a helper called during trace — example declared locals inside kernel body; helpers should work the same.

Also — the guide says control flow helpers need active device region; we're inside kernel device region. Good.

Final code:

Also consider: should I name instructions with attributes e.g. `txl.ptx.ld.global_.nc.f32`? String indexing is proven; keep.

Potential subtle problem: `txl.ptx["st.shared.v2.f32"](ptr, a, b)` — vector store with 2 values. Symmetric to v4. Fine.

Also for vec=2/4 smem staging: st_vec(STS, xs.ptr_to([base])) with base multiple of vec → shared address alignment: shared addresses are x_smem base + base*4 bytes; smem base is 16B-aligned (bank-aligned), base*4 with base even → 8B aligned for v2; base multiple of 4 → 16B for v4. Shared vector requires natural alignment — satisfied.

Wait, v2 requires 8B alignment: base even → base*4 multiple of 8. Yes. v4 requires 16B: base*4 multiple of 16 ⟺ base multiple of 4. base = (it*128+tid)*4 for vec=4 → multiple of 4. Yes.

Now finalize. Also, maybe guard `if ntiles:` unnecessary as computed ≥1.

Also — hmm, `use_smem` threshold interplay with vec: for n ≤ 12284 all fine.

One last thought on numerical stability: exp via ex2.approx — for x-mx = 0 → 2^0 = 1 exact. Sum ≥ 1. Good.

Also should I use `div.rn.f32` instead of rcp.approx for accuracy? rcp.approx.f32 max error ~1 ulp... Actually rcp.approx.f32 has max relative error 2^-23.0? Documented max ulp error is 1 for rcp.approx? PTX doc: rcp.approx.f32 max ulp error = 1.0? Something small. Fine for softmax. Alternatively use `txl.assign(inv, txl.float32(1.0) / tot)` which maps to div.rn.f32 (correctly rounded). Cost: div.rn is slower (~20 cycles) but once per row. For accuracy safety, use div.rn via expression division. Actually let me just use the expression division — simpler and definitely supported (example used `/`). But wait, will `txl.float32(1.0) / tot` lower to div.rn.f32? Presumably. Use that.

Hmm, but is division expression lowering guaranteed? Example: `txl.rsqrt(ss[0] / txl.float32(hidden_size) + ...)`. Yes division works. Use it.

Then inv = local_scalar; txl.assign(inv, txl.float32(1.0) / tot).

Final answer code below.

## content
```python
# Row-wise softmax (sm_80): one 128-thread CTA per row.
#
# Strategy (chosen at trace time from the static n):
#   * vec = 4 / 2 / 1: widest vector load whose alignment is guaranteed for
#     every row (row r starts at element offset r*n, so n % vec == 0 is
#     required for vec-wide global accesses).
#   * If the row fits in shared memory (<= 48 KiB staging), x is staged once
#     into smem; the exp values are then rewritten in place and rescaled from
#     smem (1 global read + 1 global write per element).
#   * Otherwise (very long rows) a 3-pass global variant is used: max pass,
#     exp+sum pass that parks exp(x-max) in y, and a rescale pass over y.
#
# Numerical stability: max is subtracted first (PTX max.f32 reduction), and
# exp is computed as ex2.approx.f32(x * log2e - max * log2e) via one fma.

import tirx_kernels.tirx_lite as txl

BLOCK = 128
WARPS = BLOCK // 32
FULL_MASK = 0xFFFFFFFF
LOG2E = 1.4426950408889634
NEG_INF_BITS = 0xFF800000
# 48 KiB smem budget (per-block default) minus the 16 B reduction scratch.
SMEM_LIMIT_ELEMS = 12284


def _cta_sync():
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))


def _fmax(dst, a, b):
    txl.ptx["max.f32"](dst, a, b)


def _fadd(dst, a, b):
    txl.ptx["add.f32"](dst, a, b)


def _warp_reduce(val, combine):
    # Butterfly reduction of the f32 local `val` over the 32 lanes.
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer,
            txl.reinterpret(val, ty="uint32"),
            txl.uint32(delta),
            txl.uint32(31),
            txl.uint32(FULL_MASK),
        )
        combine(val, val, txl.reinterpret(peer, ty="float32"))


def _block_reduce(val, red, combine, identity):
    # CTA-wide reduction over 4 warps: warp butterfly -> smem -> warp 0.
    # The trailing barrier separates the final broadcast load from any reuse
    # of the scratch by a following reduction.
    lane = txl.lane_id()
    warp = txl.warp_id()
    part = txl.local_scalar("float32")
    out = txl.local_scalar("float32")
    _warp_reduce(val, combine)
    with txl.if_(lane == 0), txl.then_():
        txl.ptx["st.shared.f32"](red.ptr_to([warp]), val)
    _cta_sync()
    with txl.if_(warp == 0), txl.then_():
        with txl.if_(lane < WARPS), txl.then_():
            txl.ptx["ld.shared.f32"](part, red.ptr_to([lane]))
        with txl.else_():
            txl.assign(part, identity)
        _warp_reduce(part, combine)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx["st.shared.f32"](red.ptr_to([0]), part)
    _cta_sync()
    txl.ptx["ld.shared.f32"](out, red.ptr_to([0]))
    _cta_sync()
    return out


def make_kernel(rows: int, n: int):
    # Widest vector width whose alignment holds for every row start.
    if n % 4 == 0:
        vec = 4
    elif n % 2 == 0:
        vec = 2
    else:
        vec = 1
    groups = n // vec  # exact: vec divides n
    ntiles = (groups + BLOCK - 1) // BLOCK
    use_smem = n <= SMEM_LIMIT_ELEMS

    if vec == 1:
        LDG_RO = "ld.global.nc.f32"  # read-only path for x
        LDG = "ld.global.f32"  # coherent (y is written by this kernel)
        STG = "st.global.f32"
        LDS = "ld.shared.f32"
        STS = "st.shared.f32"
    else:
        LDG_RO = "ld.global.nc.v%d.f32" % vec
        LDG = "ld.global.v%d.f32" % vec
        STG = "st.global.v%d.f32" % vec
        LDS = "ld.shared.v%d.f32" % vec
        STS = "st.shared.v%d.f32" % vec

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()

        smem = txl.smem_pool()
        red = smem.alloc([WARPS], "float32")
        xs = smem.alloc([n], "float32") if use_smem else None

        xf = txl.alloc_local([vec], "float32")
        t = txl.local_scalar("float32")
        m = txl.local_scalar("float32")
        neg_inf = txl.reinterpret(txl.uint32(NEG_INF_BITS), ty="float32")
        log2e = txl.float32(LOG2E)
        nlog2e = txl.float32(-LOG2E)

        def ld_vec(name, ptr):
            if vec == 1:
                txl.ptx[name](xf[0], ptr)
            elif vec == 2:
                txl.ptx[name](xf[0], xf[1], ptr)
            else:
                txl.ptx[name](xf[0], xf[1], xf[2], xf[3], ptr)

        def st_vec(name, ptr):
            if vec == 1:
                txl.ptx[name](ptr, xf[0])
            elif vec == 2:
                txl.ptx[name](ptr, xf[0], xf[1])
            else:
                txl.ptx[name](ptr, xf[0], xf[1], xf[2], xf[3])

        # ---- pass 1: row max (and stage x into smem when it fits) ----
        txl.assign(m, neg_inf)
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                ld_vec(LDG_RO, x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["max.f32"](m, m, xf[k])
                if use_smem:
                    st_vec(STS, xs.ptr_to([base]))

        mx = _block_reduce(m, red, _fmax, neg_inf)

        # exp(x - mx) == ex2(x * log2e + shift), shift = -mx * log2e.
        shift = txl.local_scalar("float32")
        txl.ptx["mul.f32"](shift, mx, nlog2e)

        # ---- pass 2: e = exp(x - mx), accumulate the sum ----
        s = txl.local_scalar("float32")
        txl.assign(s, txl.float32(0.0))
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                if use_smem:
                    ld_vec(LDS, xs.ptr_to([base]))
                else:
                    ld_vec(LDG_RO, x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["fma.rn.f32"](t, xf[k], log2e, shift)
                    txl.ptx["ex2.approx.f32"](xf[k], t)
                    txl.ptx["add.f32"](s, s, xf[k])
                if use_smem:
                    st_vec(STS, xs.ptr_to([base]))
                else:
                    st_vec(STG, y.ptr_to([r, base]))

        tot = _block_reduce(s, red, _fadd, txl.float32(0.0))
        inv = txl.local_scalar("float32")
        txl.assign(inv, txl.float32(1.0) / tot)

        # ---- pass 3: scale by 1/sum and store ----
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                if use_smem:
                    ld_vec(LDS, xs.ptr_to([base]))
                else:
                    ld_vec(LDG, y.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["mul.f32"](xf[k], xf[k], inv)
                st_vec(STG, y.ptr_to([r, base]))

    return softmax_txl
```
