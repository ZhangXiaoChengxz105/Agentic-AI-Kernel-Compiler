# tirx round 1

model=glm-5.3 in=7736 out=19317 cost=0.09229 latency=90.1s finish=stop effort=high usage={"completion_tokens": 19317, "completion_tokens_details": null, "prompt_tokens": 7736, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 7680, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 17028, "total_tokens": 27053}

## reasoning
We need to write a row-wise softmax kernel for sm_80, one CTA per row, block=128 threads. Correct for any n >= 1. Use warp shuffles and CTA reduction via shared memory. Vectorized loads where possible? n arbitrary; we can use float4 vectorized loads when n % 4 allows, but simpler: scalar loads with a serial loop over chunks of 128. For performance, maybe handle vectorized path via a Python-time check: since n is static per kernel trace, we can specialize: if n % 4 == 0 use float4 loads (ld.global.v4.f32), else scalar. Actually ld.global.v4.f32 in PTX takes 4 destinations. We can use txl.ptx["ld.global.v4.f32"](a,b,c,d, ptr). Store back similarly.

For exp: use txl.exp? The guide mentions `txl.rsqrt` exists in RMSNorm example. Likely `txl.exp` exists. But safer: use txl.ptx ex2? Numerical stability with x~100: exp(x - max) ∈ (0,1], so ex2.approx.f32 with scaling by log2e is fine: exp(v) = ex2(v * log2e). ex2.approx.f32 has ~2 ulp error, acceptable? "numerically stable for large |x|" — precision probably fine. Alternatively use txl.exp if available. The README mentions txl.rsqrt in the example, so txl.exp likely exists (standard TIR ops). Use txl.exp(v). Hmm, but I should avoid inventing APIs. txl.exp is a common TIR intrinsic (te.exp). In tirx-lite, txl namespace likely mirrors TIRx builder functions: txl.exp, txl.log, txl.sqrt, txl.rsqrt are standard. The example uses txl.rsqrt and txl.float32. I'll use txl.exp — reasonably safe. But to be extra safe, use PTX ex2: exp(v) = ex2.approx(v*log2(e)). That's guaranteed-correct spelling: txl.ptx.ex2.approx.f32(dst, src). ex2 with ftz? For v - max ∈ [-inf, 0], exponent can be very negative → underflow to 0, fine. Values down to exp(-large) ≈ 0, ex2.approx handles. Actually plain scalar ops: "Plain f32 scalar ops" allowed. ex2.approx.f32 is sm_80 fine.

But note max subtraction: x - max ∈ [-200, 0] for x ~ ±100. exp underflows to 0, sum ≥ 1 (since at least one element has x=max → exp(0)=1). Good.

Actually with ex2.approx.f32, is precision okay? ex2.approx has max error ~2^-22ish relative. Fine for softmax.

Alternatively use txl.exp which presumably maps to __expf or expf. I'll use PTX ex2.approx.ftz? ftz would flush denormals; exp(v) for v near 0 could be denormal? exp(-87) ≈ 1e-38 near denormal. Underflow to zero is fine for softmax. Use ex2.approx.f32 (non-ftz) to be safe.

Plan:
- grid=rows, block=128 (static).
- r = txl.cta_id()
- tid = txl.thread_id()
- Pass 1: compute row max: each thread strides through row with stride 128, keeping max; then warp reduce max via shfl butterfly (max), warp leaders store to smem, sync, warp 0 reduces 4 values, broadcast via smem + sync.
- Pass 2: sum of exp(x - max). Option: recompute loads from global (two passes over global memory) or stage x in shared memory. Shared memory: n floats; A100 has 164KB smem per block (up to 163KB with opt-in). Default 48KB without attribute. For large n (e.g., 100k), can't stage. Simplest robust: three passes over global? Pass1 max, pass2 sum, pass3 write. Or two passes with recompute of exp in pass 3: pass2 computes sum of exp; pass3 recomputes exp and divides. That's 3 global read passes. Alternatively stage into smem if n*4 <= 48KB (n <= 12288), else global re-read. To keep code simpler but performant: stage in shared memory when n <= 12288 (default 48KB limit, no dynamic smem attribute needed if allocated via smem_pool? smem_pool allocation might be static; if exceeds, compile fail). Let's implement: if n*4 <= 48*1024: stage in smem (two global reads, reads from smem afterward). Else: read global twice (pass2 and pass3 read global again).

Actually simpler and still fast: always two-pass over global with smem staging when it fits. Let me write:

SmemPool: txl.smem_pool(); x_smem = smem.alloc([n], "float32") if fits.

Vectorization: if n % 4 == 0 (and use vector loads of 4 floats = 16B, aligned since row-major with row length n multiple of 4 → each row start offset r*n*4 bytes multiple of 16). Use ld.global.v4.f32 / st.global.v4.f32 / ld.shared.v4.f32 / st.shared.v4.f32. If n%4 != 0, could still vectorize partially... keep it simple: scalar path otherwise. Actually many n are multiples of 4; but to be correct for any n, need fallback. Alternatively use vector width 1 always with n arbitrary — simpler but slower. Let's do Python-level specialization: vec = 4 if n % 4 == 0 else 1. Handle both with generic code parameterized by vec? Vector PTX ld takes 4 dest regs; scalar takes 1. Write two branches in Python (if vec==4 vs vec==1) sharing helper structure.

Hmm, also could handle n%2==0 with v2. Let's do vec ∈ {4,2,1} using ld.global.v2.f32 etc. That adds complexity. I'll support vec = 4 if n%4==0 else (2 if n%2==0 else 1). Use instruction names as strings via txl.ptx["ld.global.v4.f32"].

Indexing: with vec, each thread handles elements at positions p = tid*vec + iter*128*vec + k for k in range(vec). Number of vec-chunks: nvec = n // vec, exactly divisible. Loop iterations: ceil(nvec / 128).

Alignment: base pointer alignment: torch tensors are 256B aligned typically, row offset r*n*4 bytes — multiple of 4*vec bytes since n%vec==0. Good.

Max reduction:

def warp_reduce_max(v): butterfly with shfl.sync.bfly.b32 on reinterpreted uint32, take max via txl.max? Is txl.max available? PTX max.f32: txl.ptx.max.f32(dst, a, b). Use that — safe.

txl.ptx.shfl_sync.bfly.b32(peer, reinterpret(v,"uint32"), delta, 31, FULL_MASK) — as in the example.

Warp count = 4 (block 128). smem for partials: smem.alloc([4], "float32").

bar.sync: txl.ptx.bar.sync(txl.uint32(0), txl.uint32(128)).

For max: after warp reduce, lane0 writes to red_smem[warp]; bar.sync; threads with tid<4 read red_smem[tid], reduce among first warp (butterfly again over 4 lanes? butterfly over full warp with max works since max over 32 lanes includes the 4 values padded... but other lanes read garbage). Better: have warp 0 lanes 0..3 load values, others load -inf, butterfly max, lane0 writes result to red_smem[0]; bar.sync; all read red_smem[0]. Like the RMSNorm sum pattern.

Alternatively, all threads read all 4 smem values directly: after storing warp maxima, bar.sync, then each thread reads red_smem[0..3] and computes max locally — 4 smem loads per thread, cheap and simple, avoids second butterfly. I'll do the warp-0 butterfly approach like the example for elegance... Actually direct read of 4 values by every thread is simpler and fine. But per-thread 4 loads + 3 max ops. Fine. Hmm, but I need it as a scalar in registers for subsequent passes; every thread needs the max. Reading 4 smem floats and maxing: m = max(max(a,b),max(c,d)). Simple. Same for sum.

Actually let me just do warp0 reduction to smem[0] and broadcast — the canonical pattern. Either works. I'll do the "all threads read 4 partials" approach: fewer instructions total code complexity.

Wait — careful: with 128 threads, num_warps=4. block=128 always in spec. Good, nwarp=4.

Pass structure:

if stage_in_smem (n <= SMEM_CAP_ELEMS where cap = 48*1024//4 = 12288):
  Pass 1: load from global (vectorized), store to smem, track max.
  Reduce max.
  Pass 2: read smem, compute e = ex2((x - max)*log2e), accumulate sum, (optionally store e into smem to avoid recompute — overwrite x_smem with e; then pass 3 reads e from smem, divides by sum, stores). Storing e into smem saves exp recompute. Do that.
  Reduce sum.
  Pass 3: read smem e, multiply by 1/sum (use txl.rcp? or division; use `txl.ptx.rcp.approx.f32`? safer: y = e / sum via fdiv... just use e * (1.0/sum) computed once per thread: inv = txl.float32? Division: `e / sum` using txl division operator? In DSL, `/` on expressions likely maps to fdiv. The example uses `ss[0] / txl.float32(hidden_size)`. So `/` works. Use `inv = txl.float32(1.0) / total` then multiply, or per-element division — multiply by reciprocal is faster. Precision fine.
  Store to global (vectorized).

else (large n, no smem staging):
  Pass 1: global load, track max.
  Reduce max.
  Pass 2: global load, e = exp(x - max), sum += e.
  Reduce sum.
  Pass 3: global load again, y = exp(x-max)/sum, store.

That's 3 global reads for large n; acceptable. Alternatively for large n could use smem up to 48KB with tiles — but n > 12288 rows are rare; keep simple.

Actually, we could raise smem via kernel attrs... skip.

Also note: smem.alloc requires device region; sizes static (n is Python int at trace time). Good.

Vectorized load with vec=4: pointer x.ptr_to([r, col]) where col = base + k for k in 0..3; we load v4 into 4 locals, then store each. With smem staging also st.shared.v4.f32.

Let me now think about txl syntax details from examples:

- txl.local_scalar("float32") creates writable scalar; txl.assign(dst, src).
- txl.alloc_local([len], "float32") register array.
- smem tensor: smem.alloc([n], "float32"), ptr via x_smem.ptr_to([i]).
- txl.ptx.ld.global.f32(dst, ptr) — DPS with dest first.
- txl.ptx["ld.global.v4.f32"](d0,d1,d2,d3, ptr).
- shfl: txl.ptx.shfl_sync.bfly.b32(peer, u32val, delta, 31, mask).
- max: txl.ptx.max.f32(dst, a, b). Check PTX name: `max.f32` — yes. For max reduction init: -inf. Use txl.float32("-inf")? Maybe. Or a very negative number like -3.402823466e38. Safer: use txl.ptx.mov.b32 with 0xFF7FFFFF bits? Could use txl.reinterpret(txl.uint32(0xFF7FFFFF), ty="float32"). Simplest: since pass1 always loads at least... every thread may load zero elements? n>=1, but with vec chunks, threads with tid*vec >= nvec load nothing in all iterations (e.g., n=1: only thread 0 loads). So init max to -inf needed. Use float("-inf") constant: txl.float32(float("-inf")) — does the DSL support inf literal? Might emit arith.constant infinity; TIR supports inf float. Reasonably safe. Alternative: 0xFF7FFFFF reinterpret. I'll use reinterpret of uint32 0xff7fffff to be robust:

NEG_INF = txl.reinterpret(txl.uint32(0xFF7FFFFF), ty="float32") — txl.reinterpret(value, ty=...) per README: `txl.reinterpret(value, ty=txl.uint32)`. So ty can be a string "float32" presumably. The example used ty="uint32" string. OK.

Hmm, but careful: reinterpreting inside a loop each iteration — hoist into a local? Assign to local_scalar once. Or just use Python constant float("-inf") with txl.float32 — I think txl.float32(float('inf')) works in TVM (FloatImm supports inf). I'll use the reinterpret route in an init via txl.assign(m, ...) where m is local_scalar, computed once at trace... Actually expressions are traced; do it at top: 

neg_inf = txl.reinterpret(txl.uint32(0xFF7FFFFF), ty="float32")

Then txl.assign(mval, neg_inf).

Alternatively use max with initial value from first load... complexity. Keep -inf.

Actually simpler: since every thread's local max only matters via reduction, and threads with no elements contribute -inf, that's fine with max.

For sum init 0.0.

ex2: exp(v) = ex2(v * 1.4426950408889634). Compute e = ex2((x - m) * LOG2E). Instruction: txl.ptx.ex2.approx.f32(dst, src). Name with dots: txl.ptx.ex2.approx.f32 — attribute chain works like txl.ptx.ld.global_.f32 (note "global_"). For ex2.approx.f32, attribute access: txl.ptx.ex2.approx.f32? Python attribute access requires valid identifiers: "ex2", "approx", "f32" all valid. ld.global_.f32 uses underscore because "global" is a reserved... they renamed. So txl.ptx.ex2.approx.f32 should work. Alternatively txl.ptx["ex2.approx.f32"]. Use bracket form for safety: txl.ptx["ex2.approx.f32"](e, src). The RMSNorm example uses bracket form LD_G_V4 = "ld.global.v4.b32". Good, bracket form is safe.

Also for max: txl.ptx["max.f32"](dst, a, b). DPS: dest first.

Now precision: ex2.approx relative error ~2^-22; softmax typical tolerance fine.

Alternatively use txl.exp if it exists... risk. Use ex2.

Now the reduce helpers:

def warp_bfly_max(v):  # v is local_scalar
    peer = txl.local_scalar("uint32")
    for delta in (16,8,4,2,1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(v, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF))
        txl.ptx["max.f32"](v, v, txl.reinterpret(peer, ty="float32"))

Wait DPS: txl.ptx["max.f32"](v, v, other) — dest is v, sources v and other. Fine.

Hmm: txl.reinterpret(v, ty="uint32") where v is a local_scalar — local_scalar probably acts as expression (like acc[0] in example, used directly in expressions). Yes, in the example acc[0] used directly. local_scalar itself — I think local_scalar is a handle used like a variable; in example, `rms` local_scalar used as `txl.cuda.make_float2(rms, rms)` and in expressions. And assigned via txl.assign(rms, ...). So local_scalar used directly as expression. But indexing? For alloc_local arrays use ss[0]. For local_scalar use directly. OK.

CTA reduce max:
- warp_bfly_max(m)
- lane = tid & 31; warp = tid >> 5
- if lane==0: st.shared.f32 red_smem[warp] = m
- bar.sync(0,128)
- all threads: read red_smem[0..3] into locals and max → cta_max. Or warp0 pattern. I'll do: each thread loads 4 values and maxes:

mval = red[0]; for w in 1..3: max.f32(mval, mval, red_w). Need ld.shared.f32 into temp local each time. Write:

tmp = txl.local_scalar("float32")
txl.ptx.ld.shared.f32(tmp, red_smem.ptr_to([0])) — wait DPS: ld.shared.f32(dst, ptr). In example: txl.ptx[LD_S_F32](ss[0], sum_sq_smem.ptr_to([lane])). Yes.

Then for w in 1..3: txl.ptx.ld.shared.f32(tmp2, red_smem.ptr_to([w])); max.f32(mval, mval, tmp2). Reuse tmp.

Similarly for sum with add.f32 or just `+` operator (example used `acc[0] + reinterpret(peer,...)`). Use +.

Combine: I could write one generic CTA reduce function parameterized by warp op. Write two separate small helpers for clarity.

Then broadcast: since every thread computes it from smem, no extra broadcast needed. 

Sum: same with add.

Now passes. Define:

BLOCK = 128
NWARP = 4
vec chosen.
nvec = n // vec
iters = ceildiv(nvec, BLOCK)

Vector registers: xv = txl.alloc_local([vec], "float32") for load staging; for vec==4 use txl.ptx["ld.global.v4.f32"](xv[0],xv[1],xv[2],xv[3], x.ptr_to([r, base])). For vec==2: "ld.global.v2.f32". vec==1: ld.global.f32.

Index type: ptr_to([r, col]) — r is cta_id (int32?), col int32 expression. Tensor shape (rows, n) with Python ints → static shape? txl.Tensor((rows, n), "float32") with ints — static extents. ptr_to indices: need int32/int64 expressions. In scale2 example, `x.ptr_to([r, col])` with r=txl.cta_id() and col int expression. Fine.

col = it * (BLOCK*vec) + tid*vec, an int32 expression. Guard: col < n (n static int — comparison with Python int probably fine, or wrap txl.int32(n)). In scale2: `col < n` with n Python int from closure. OK.

Pass 1 (max):
mval = local_scalar float32, init -inf.
with txl.serial(iters) as it:
    base = it * (BLOCK*vec) + tid * vec   # first element index of this thread's vector
    with txl.if_(base < n), txl.then_():
        load vec floats from global x[r, base..base+vec-1]
        if stage: store to smem x_smem[base..]
        for k: max.f32(mval, mval, xv[k])

Wait if stage, we need x values later from smem — but also need them for pass 2 reading from smem. Fine.

But when staging, storing to smem: vec==4 → st.shared.v4.f32(x_smem.ptr_to([base]), xv[0..3]); vec==2 → v2; vec==1 → st.shared.f32.

Pass 2 (sum of exp):
sval = local_scalar float32 init 0.
with txl.serial(iters) as it:
    base = ...
    if base < n:
        load vec floats (from smem if stage else global)
        for k: e = ex2((v - mval)*LOG2E); sval += e; if stage: store e back to smem (overwrites x with e — saves recompute in pass3).

Store back to smem in pass 2: need e in registers; store via st.shared (v4 if vec 4). But careful: pass2 reads smem then writes same location — within a thread, read then write own locations; no cross-thread hazard. Fine.

Pass 3 (normalize):
inv = 1.0 / total (after sum reduce)
with txl.serial(iters) as it:
    base...
    if base < n:
        load e values (smem if stage else recompute from global: load global, e=ex2((x-m)*log2e))
        y[r, base+k] = e * inv via st.global (v4 etc.)

For non-staged large-n path, pass 3 loads global a third time. OK.

Hmm wait — for the staged path, could skip pass-3 smem load recompute... it's already reading smem, minimal.

Edge: nvec==0 impossible since n>=1, vec divides n so nvec>=1. iters >= 1.

Guard `base < n`: base is it*BLOCK*vec + tid*vec; since vec divides n, base<n ⟺ base/vec < nvec. fine.

Shared memory for red partials: red_smem = smem.alloc([NWARP], "float32"). When staging, x_smem = smem.alloc([n], "float32").

Barrier: txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK)). Define cta_sync().

Now Python-level branching to emit either staged or non-staged code. Write helper functions inside make_kernel or inside the kernel body using Python if.

Note: helpers defined inside the traced function are fine (RMSNorm defines helpers inside).

Careful: local_scalar alloc must be inside device region — yes, everything after device_entry flat.

Also note: txl.local_scalar("float32") — per example `ss`? Example: `pk = txl.local_scalar("uint64")`, `rms = txl.local_scalar("float32")`. Good.

Assign: txl.assign(mval, neg_inf_expr). For -inf, I'll use txl.float32(float("-inf")). Risk: constant creation with inf. TVM FloatImm supports inf. I'll go with reinterpret to be safe:

neg_inf: txl.reinterpret(txl.uint32(0xFF7FFFFF), ty="float32")? That's actually -FLT_MAX, not -inf. max with -FLT_MAX works fine for reduction (any real x > -FLT_MAX; x could be -inf? x is finite per problem statement "x ~ 100"). Use -FLT_MAX bits 0xFF7FFFFF. Good, avoids inf entirely.

But wait, if a thread loads no elements, contributes -FLT_MAX; row max is real value. fine.

ex2 src: (x - mval) * LOG2E. mval is local_scalar. Fine.

Now, is `txl.ptx.shfl_sync.bfly.b32` correct spelling? From example: txl.ptx.shfl_sync.bfly.b32(peer, reinterpret, delta, 31, FULL_MASK). Yes.

max.f32 DPS: txl.ptx.max.f32? "max" is a valid Python identifier... but txl.ptx.ld.global_.f32 shows `global` got underscore (reserved keyword in Python? "global" is a keyword — yes). "max", "min", "add" are not keywords (add no, max no). But to be safe use bracket strings: txl.ptx["max.f32"]. PTX instruction `max.f32 d, a, b` exists. Good.

Actually, could I just use Python max via... no. Use PTX max.

Also could use txl.max if it exists in TIR ops (TIR has max). Unknown; PTX is safer.

Division: `txl.float32(1.0) / total` — `/` operator on expressions; example used `ss[0] / txl.float32(hidden_size)`. Good.

Now vector instruction names:
- "ld.global.v4.f32", "ld.global.v2.f32", "ld.global.f32"
- "st.global.v4.f32", "st.global.v2.f32", "st.global.f32"
- "ld.shared.v4.f32" etc., "st.shared.v4.f32".

From example LD_S_V4 = "ld.shared.v4.f32", ST_G_V4 = "st.global.v4.b32" (they used b32 for global store of packed words; for float we can use f32 variant — does PTX have st.global.v4.f32? Yes, vector stores support .f32: st.global.v4.f32 is valid PTX). Similarly ld.global.v4.f32 valid.

Alignment requirement: 16B for v4. Row start r*n elements; n%4==0 → 16B aligned if base is. Torch allocations are 256B+ aligned. Good. v2 needs 8B: n%2==0.

Shared v4 also needs 16B alignment of x_smem + base*4; smem alloc alignment — smem_pool allocations likely aligned to at least... hmm, if we allocate red_smem [4] floats first then x_smem, offset 16 bytes → aligned 16B only if pool base aligned appropriately. To be safe, allocate x_smem first (large), then red_smem. Even then, is the first allocation 16B-aligned? Likely pool base is aligned. I'll allocate x_smem first, red_smem second. Actually order of allocation in pool determines offsets; allocate x_smem first.

Hmm, but if n%4==0, x_smem size n*4 bytes is multiple of 16 (well, 4*n; n%4==0 → multiple of 16). Wait, vec==4 only when n%4==0, and staging path requires smem; if n%4 != 0, vec is 2 or 1, needing 8B or 4B alignment — n*4 bytes multiple of 8 when n%2==0. So x_smem first, red_smem second always gives aligned x_smem (assuming pool base 16B aligned, typical). Good.

Also for safety could just use scalar shared accesses... but vector is faster. Keep vector with the ordering.

Now, smem capacity: default static shared memory limit without opt-in is 48KB. n <= 12288 floats = 49152 bytes... exactly 48KB is allowed as static? Static shared memory limit is 48KB (49152 bytes) per block without attribute. n*4 <= 49152 → n <= 12288. Plus red_smem 16 bytes → total 49168 > 49152 for n=12288! So use n <= 12288 - 4, i.e., stage if n*4 + 16 <= 49152 → n <= 12284. Let's set SMEM_ELEMS = (48*1024 - 64) // 4 to leave slack. Actually simpler: stage if n <= 12000 (some margin). I'll compute: stage = n * 4 <= 48*1024 - 256. Fine.

Hmm, but also smem_pool might support larger with dynamic smem attributes via kernel_attrs... not needed.

Now also consider: when not staging (huge n), pass 2 and pass 3 both read global. Fine.

Let me also double-check `txl.serial(count)` usage: `with txl.serial(n_tiles) as ki:` — ki is loop var expression. Inside, `st = (ki * nthreads + tid) * vec`. Good.

`txl.if_` syntax: `with txl.if_(cond):` then `with txl.then_():`. In scale2 example: `with txl.if_(col < n), txl.then_():` — combined form. Both fine.

Loop var arithmetic: ki * (BLOCK * vec) + tid * vec — int32.

Now write helper for vector global load:

def ld_global(dst, ptr):  # dst is alloc_local array of length vec
    if vec == 4: txl.ptx["ld.global.v4.f32"](dst[0], dst[1], dst[2], dst[3], ptr)
    elif vec == 2: txl.ptx["ld.global.v2.f32"](dst[0], dst[1], ptr)
    else: txl.ptx.ld.global_.f32(dst[0], ptr)

Wait txl.ptx.ld.global_.f32 — that's the dotted form (valid because of underscore). Or bracket "ld.global.f32". Use bracket uniformly.

Similarly st_global(vals, ptr), ld_shared, st_shared.

Note: in the example, DPS ld: txl.ptx[LD_G_V4](iw[0], iw[1], iw[2], iw[3], ptr). Good.

Now the reduce for max using local_scalar mval — but warp reduce via shfl needs reinterpret of mval: txl.reinterpret(mval, ty="uint32"). mval is local_scalar handle — does reinterpret accept it? Example reinterpreted acc[0] (alloc_local element). local_scalar `rms` was used in arithmetic directly. reinterpret(rms,...) should also work since it's an expression. I'll use alloc_local([1], "float32") arrays instead of local_scalar to exactly match example usage patterns (ss[0]). Actually local_scalar is described as "writable scalars use txl.local_scalar". Both fine. I'll use local_scalar and use it directly like `rms`. For shfl reinterpret, txl.reinterpret(m, ty="uint32") — should work.

Hmm, to minimize risk, use alloc_local([1], "float32") style like `ss` in the example: assigned with txl.assign(ss[0], ...)? Wait, in the example ss is alloc_local([1],"float32") and used as ss[0] with txl.assign(ss[0], ss[0] + ...) — wait example: `txl.assign(ss[0], ss[0] + xf[kv] * xf[kv])`. Yes alloc_local element assignment via txl.assign. And local_scalar assigned via txl.assign(rms, ...). Both patterns shown. I'll use local_scalar (cleaner) since it's shown: `txl.assign(row, bx)`, `txl.assign(rms, ...)`, and used in expressions `make_float2(rms, rms)`, `row * hidden_size`. So local_scalar works as expression. 

For the shfl reinterpret of a local_scalar — not shown, but reinterpret takes any expression. Should be fine. Alternatively, keep per-warp accumulators as local_scalar and reinterpret — risk minimal.

Actually, one concern: txl.reinterpret signature: `txl.reinterpret(value, ty=txl.uint32)`. In the example, they called `txl.reinterpret(acc[0], ty="uint32")` with string. And reinterpret back: `txl.reinterpret(peer, ty="float32")`. Good.

Now the CTA max helper:

def cta_reduce(mval, is_sum... ) — write two functions:

def warp_max(v):
    peer = txl.local_scalar("uint32")
    for d in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(v, ty="uint32"), txl.uint32(d), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.ptx["max.f32"](v, v, txl.reinterpret(peer, ty="float32"))

Hmm wait, DPS dest: max.f32 d, a, b → txl.ptx["max.f32"](v, v, other). But is v writable as a DPS destination? local_scalar should be (rms was assigned). Example used mov with pk (local_scalar) as dest: txl.ptx.mov.b32(dst[...],...) and `txl.ptx[MUL_F32X2](pk, a, b)` with pk local_scalar as dest. So yes.

def cta_max(v):
    warp_max(v)
    lane = tid & 31  # these are expressions; compute once outside
    warp = tid >> 5
    with txl.if_(lane == 0), txl.then_():
        txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), v)
    cta_sync()
    # every thread reduces the 4 partials
    t = txl.local_scalar("float32")
    txl.ptx["ld.shared.f32"](t, red_smem.ptr_to([0]))
    for w in range(1, NWARP):
        txl.ptx["ld.shared.f32"](t2? ...)

I need two temps or reuse: load into tmp then max into v:
    tmp = txl.local_scalar("float32")
    txl.ptx["ld.shared.f32"](v, red_smem.ptr_to([0]))  # overwrite v
    for w in 1..3:
        txl.ptx["ld.shared.f32"](tmp, red_smem.ptr_to([w]))
        txl.ptx["max.f32"](v, v, tmp)

Similarly cta_sum with add: txl.ptx["add.f32"](v, v, tmp) or v + tmp then assign. Use assign with + (shown in example). 

warp_sum:
    peer = local_scalar uint32
    for d in (16,8,4,2,1):
        shfl; txl.assign(v, v + reinterpret(peer,"float32"))

Both helpers defined inside kernel body (device region) — they emit code when called. But defining them before use inside traced function: the def statements are Python, fine.

lane/warp computed once: lane = tid & 31; warp = tid >> 5. In example: `lane = tid & 31`, `warp = tid >> 5`. Good.

cta_sync: txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK)).

Now main flow. Let me draft:

```python
import math
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
BLOCK = 128
NWARP = BLOCK // 32
LOG2E = 1.4426950408889634
NEG_INF_BITS = 0xFF7FFFFF  # -FLT_MAX
SMEM_CAP_BYTES = 48 * 1024 - 256


def make_kernel(rows: int, n: int):
    vec = 4 if n % 4 == 0 else (2 if n % 2 == 0 else 1)
    nvec = n // vec
    iters = (nvec + BLOCK - 1) // BLOCK
    stage = n * 4 <= SMEM_CAP_BYTES

    if vec == 4:
        LD_G = "ld.global.v4.f32"; ST_G = "st.global.v4.f32"
        LD_S = "ld.shared.v4.f32"; ST_S = "st.shared.v4.f32"
    elif vec == 2:
        ...
    else:
        ...

    @txl.kernel()
    def softmax(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        if stage:
            x_smem = smem.alloc([n], "float32")
        red_smem = smem.alloc([NWARP], "float32")
```

Wait — Python `if stage:` inside traced function: fine, it's Python control flow at trace time. But the `if` must be at Python level, and defining x_smem conditionally; later references need to know stage too. OK since stage is trace-time.

Hmm — careful: smem.alloc alignment. If stage, x_smem first then red_smem. Good.

Continue:

```python
        xv = txl.alloc_local([vec], "float32")

        def ld_g(ptr):
            if vec == 4:
                txl.ptx[LD_G](xv[0], xv[1], xv[2], xv[3], ptr)
            elif vec == 2:
                txl.ptx[LD_G](xv[0], xv[1], ptr)
            else:
                txl.ptx[LD_G](xv[0], ptr)
```

Wait but if vec==1, LD_G = "ld.global.f32". OK.

But hold on: does x.ptr_to([r, base]) with r=cta_id produce correct addressing with static shape? scale2 did exactly this. Yes.

Helpers:

```python
        def st_s(ptr):
            if vec == 4: txl.ptx[ST_S](xv[0],...,ptr)
            ...
        def ld_s(ptr): ...
        def st_g(ptr): ...
```

But pass 2 stores computed e values into xv then st_s. Fine.

Now code:

```python
        m = txl.local_scalar("float32")
        s = txl.local_scalar("float32")
        e_tmp? not needed

        # --- pass 1: max (+ stage into smem) ---
        txl.assign(m, txl.reinterpret(txl.uint32(NEG_INF_BITS), ty="float32"))
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                ld_g(x.ptr_to([r, base]))
                if stage:
                    st_s(x_smem.ptr_to([base]))
                for k in range(vec):
                    txl.ptx["max.f32"](m, m, xv[k])
```

Wait: `it * (BLOCK*vec)`: it is loop var; BLOCK*vec Python int. OK.

Reduction of m:

```python
        warp_max(m)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), m)
        cta_sync()
        tmp = txl.local_scalar("float32")
        txl.ptx["ld.shared.f32"](m, red_smem.ptr_to([0]))
        for w in range(1, NWARP):
            txl.ptx["ld.shared.f32"](tmp, red_smem.ptr_to([w]))
            txl.ptx["max.f32"](m, m, tmp)
```

Hmm — wait, warp_max(m) leaves lane0 with the warp max; all lanes have it actually (butterfly gives all lanes the full reduction). Yes, butterfly gives all lanes the total. So lane==0 store is fine.

Note: after cta reduce of max, no barrier needed before pass 2 writes to smem? Pass 2 (staged path) reads x_smem written in pass 1 by other threads → need bar.sync after pass 1 stores before pass 2 reads. The cta_sync inside cta_max handles it (bar.sync after warp leaders store). But wait: is there a hazard: pass-2 st_s writes to x_smem locations read by other threads in pass 2? In staged path pass 2, each thread reads its own locations and writes e back to same locations. But another thread never reads those. However pass-2 stores to x_smem could race with pass-2 reads by other threads of *different* locations — no overlap. But there's a subtle issue: thread A writes x_smem[i] (its own) while thread B reads x_smem[j], j≠i — fine.

But across iterations within pass 2, no cross-thread access. OK. However, we need a bar between the red_smem reads (in cta_max tail) and later reuse of red_smem in cta_sum — cta_sum starts with warp butterfly (no smem), then writes red_smem after... thread could write red_smem[warp] while another thread still reading red_smem in the max tail? The max tail reads happen right after first bar.sync; then pass 2 runs (long), then warp_sum, then lane0 stores red_smem. Between max-tail reads and sum stores there's the whole pass 2 — but no barrier! Pass 2 length varies per thread? All threads execute same loop trip count (iters static), but the guarded body varies. Still, could a thread reach the sum store while another hasn't finished reading red partials from max? The max-tail reads (4 ld.shared) happen immediately after bar.sync for all threads; pass 2 involves loads from global/smem. In theory without a barrier, thread A could race ahead through pass2 (all its loads hit cache) and store red_smem[warp] for sum while thread B is still... no wait, thread B's max-tail reads happen before thread B's pass 2; thread A's sum-store happens after thread A's pass 2. Could A's sum-store precede B's max-tail read? A must complete its own max-tail reads and full pass 2 first. B just needs to do 4 smem loads. For A to overtake B's 4 loads, B would have to be stalled ~ hundreds of cycles. Possible in theory (no forward-progress guarantees within warp divergence... actually threads in different warps can be arbitrarily delayed). To be safe, add cta_sync() after the max-tail reads (end of cta_reduce) before pass 2. Similarly after sum-tail reads before pass 3 stores to global — pass 3 only reads smem (staged) or global; and pass 3 doesn't touch red_smem again. But kernel ends; no further reuse. Also staged path: pass 2 stores e to x_smem; pass 3 reads x_smem — need bar between pass 2 stores and pass 3 reads (cross-thread? each thread reads only its own locations in pass 3, same as pass 2). Own locations only → no cross-thread hazard, no bar needed. But wait — pass 2's own-location store then pass 3 own-location load by same thread: fine.

Hmm, actually is it truly own locations only? base = it*BLOCK*vec + tid*vec — each (it, tid) maps to distinct element ranges, and the same tid covers the same base set in each pass. Yes, each thread reads exactly the locations it wrote. So no barrier needed between pass 2 and 3 for the staged path.

For the non-staged path, pass 3 re-reads global — fine.

But the red_smem reuse hazard: add cta_sync at end of each reduce helper. Cheap. Also note the RMSNorm example uses bar.sync between phases similarly.

Let me restructure cta reduce as a single function emitting: warp butterfly, store, sync, read+reduce, sync. Two variants (max/sum). Write:

```python
        def cta_max(v):
            warp_max(v)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), v)
            cta_sync()
            txl.ptx["ld.shared.f32"](v, red_smem.ptr_to([0]))
            for w in range(1, NWARP):
                txl.ptx["ld.shared.f32"](tmp, red_smem.ptr_to([w]))
                txl.ptx["max.f32"](v, v, tmp)
            cta_sync()
```

Wait but tmp defined where? Define tmp inside function each call — creates multiple locals, fine. Or define once outside. Define once outside as `red_tmp = txl.local_scalar("float32")`.

Hmm, one problem: `txl.ptx["ld.shared.f32"](v, ...)` — v is local_scalar, DPS dest. The example used ss[0] (alloc_local element) as ld dest. mov.b32 used pk (local_scalar) as dest. Should be OK.

Similarly cta_sum with `txl.assign(v, v + red_tmp)`.

Also there's a subtlety with warp_sum: after butterfly all lanes have sum. lane==0 stores.

Now pass 2:

```python
        # --- pass 2: sum of exp(x - max) ---
        txl.assign(s, txl.float32(0.0))
        l2e = txl.float32(LOG2E)  # hoist? It's a constant; fine inline
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                if stage:
                    ld_s(x_smem.ptr_to([base]))
                else:
                    ld_g(x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["ex2.approx.f32"](xv[k], (xv[k] - m) * txl.float32(LOG2E))
                    txl.assign(s, s + xv[k])
                if stage:
                    st_s(x_smem.ptr_to([base]))
```

ex2.approx DPS dest xv[k] (alloc_local element as dest — example used mov.b32(dst[i], ...) so fine).

Wait, DPS with dest also a source? `(xv[k] - m) * c` is a separate expression register; ex2 writes xv[k]. Fine.

Pass 3:

```python
        cta_sum(s)
        inv = ... compute: txl.assign(s, txl.float32(1.0) / s)  # reuse s as inv
        with txl.serial(iters) as it:
            base = ...
            with txl.if_(base < n), txl.then_():
                if stage:
                    ld_s(x_smem.ptr_to([base]))
                else:
                    ld_g(x.ptr_to([r, base]))
                    for k in range(vec):
                        txl.ptx["ex2.approx.f32"](xv[k], (xv[k] - m) * txl.float32(LOG2E))
                for k in range(vec):
                    txl.assign(xv[k], xv[k] * s)
                st_g(y.ptr_to([r, base]))
```

Hmm — for the non-staged path, this is the third global read. Acceptable.

Wait, `txl.float32(1.0) / s` — division maps to fdiv (slow full-precision div, one per thread, negligible).

Numerical check: sum s >= 1 (since max element gives exp(0)=1). ex2.approx of 0*log2e = ex2(0) = 1 exactly. Good.

Precision concern: ex2.approx.f32 — "approximate exponential" with max relative error about 2^-22... Actually documented max ulp error for ex2.approx.f32 is 2 ulp. Fine.

Should I instead use exact exp? PTX ex2.approx.ftz? Non-ftz version handles denormals? For very negative arguments result underflows to 0 — fine either way. Use non-ftz "ex2.approx.f32".

Now, potential correctness issue: `(xv[k] - m)` where xv[k] is -FLT_MAX possible and m is -FLT_MAX (row all -FLT_MAX) → 0 → exp(0)=1 → uniform. fine.

Another consideration: the guard `base < n` — base is expression `it * 128*vec + tid*vec`. it: loop var of serial with static extent — type int32 presumably. tid int32. Comparison with Python int n — the scale2 example did `col < n`. Good.

Index into tensor ptr_to([r, base]): r int32, base int32. Static shapes (rows, n) as Python ints. Should be fine (scale2 identical).

Edge case: n=1 → vec=1, nvec=1, iters=1. Thread 0 loads x[r,0]. Others skip. Max = x[r,0] via reduction. Sum=1. y=1. Good.

Edge case n=2: vec=2, thread 0 loads v2. iters=1. Good.

n=130: vec... 130%4=2, %2=0 → vec=2, nvec=65, iters=1. Threads 0..64 load; thread tid has base=2*tid < 130 for tid<=64. Good.

n=5: vec=1, nvec=5, iters=1, tid<5 load.

Large n staged: n=12288: 12288*4=49152 > SMEM_CAP (48*1024-256=48896) → not staged. Hmm, that loses staging for 12288. Set cap = 48*1024 - 64 = 49152-64; need n*4 + 16 (red) <= 49152 → n <= 12284. Let me just define SMEM_ELEMS = (48 * 1024 - 64) // 4 = 12280. stage = n <= 12280.

Actually, does the pool allocate exactly? x_smem n floats + red 16 bytes. For n=12280: 49120+16=49136 ≤ 49152. OK.

One more: does smem_pool require total size declared or handle it? smem.alloc presumably appends. Fine.

Now `txl.alloc_local([vec], "float32")` — vec is Python int ≥ 1. Good.

Also, the loop `for k in range(vec)` with vec possibly 1: fine.

Now instruction names for vec==1: LD_G="ld.global.f32" — is that a valid PTX name? Yes (ld.global.f32). Shared: "ld.shared.f32", "st.shared.f32" (example used those exact strings LD_S_F32 = "ld.shared.f32").

Now warp helpers. FULL_MASK = 0xFFFFFFFF. shfl args: (dest, src, delta, clamp, mask). Example: txl.ptx.shfl_sync.bfly.b32(peer, val, delta, 31, FULL_MASK). Good.

Define:

```python
        def warp_max(v):
            peer = txl.local_scalar("uint32")
            for d in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(v, ty="uint32"),
                    txl.uint32(d),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.ptx["max.f32"](v, v, txl.reinterpret(peer, ty="float32"))
```

Hmm — wait, `txl.reinterpret(v, ty="uint32")` where v is a local_scalar handle. reinterpret expects an expression; a local scalar variable read... In the example they reinterpreted acc[0] and peer (peer is local_scalar!) — `txl.reinterpret(peer, ty="float32")` in warp_sum. Yes! peer is local_scalar and it's reinterpreted. So local_scalar works in reinterpret. 

Also potential issue: DPS max with dest = same as src a: `max.f32 v, v, tmp` fine at PTX level.

cta_sync:

```python
        def cta_sync():
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
```

Now one concern: helpers referencing `lane`, `warp`, `tmp`, `red_smem` from enclosing scope — closure over traced values. The RMSNorm example's helpers do the same (cta_sync references nthreads). Fine.

Another concern: `warp = tid >> 5` — tid is expression; >> works (example). `lane = tid & 31` (example).

Now putting the staged/non-staged branching: I'll use Python if inside the traced body:

```python
        if stage:
            ...pass code with smem...
        else:
            ...
```

But passes are interleaved with reductions. I'll write per-pass branches or a helper `load_x(base)` that loads into xv from smem or global:

```python
        def load_x(base):
            if stage:
                ld_s(x_smem.ptr_to([base]))
            else:
                ld_g(x.ptr_to([r, base]))
```

And pass1 uses ld_g + st_s specially.

Let me also double-check `smem.alloc` signature: example `x_smem = smem.alloc([hidden_size], "float32")`. Good.

Also — `txl.smem_pool()` then `.alloc`. Good.

Potential IR-contract issue: "Every kernel is checked against the low-level IR contract." Things like reading uninitialized registers? We always write xv before read within guards. m initialized. Fine.

One more concern: in pass 3 non-staged path, I compute ex2 into xv[k] inside the guard. Fine.

Let me also consider whether `txl.ptx["ex2.approx.f32"]` DPS is (dest, src). Yes, ex2.approx.f32 d, a.

Also, should I use `mul.f32` explicitly? `(xv[k] - m) * txl.float32(LOG2E)` — arithmetic operators shown in example (`xf[kv] * xf[kv]`, `ss[0] + ...`). Fine.

Constant folding: LOG2E constant per element — compiler handles.

Now think about whether there could be an issue with `it * (BLOCK * vec)` when iters is a Python int: `with txl.serial(iters) as it:` — example used serial(n_tiles) with Python int. Good.

Performance thoughts: two global read passes (staged path: pass1 read global; pass2,3 from smem; one global write). Good. Non-staged: three global reads + one write. OK.

Could improve: use __ldg / non-coherent? default ld.global is fine (read-only data might use ld.global.nc — skip).

Now, one subtle thing: guard `base < n` — but for vec>1, thread's vector spans base..base+vec-1 all < n since vec | n. Good.

Alignment of x_smem vector accesses: x_smem base offset 0 in pool (allocated first). Assume 16B aligned. red_smem second. For vec=2, need 8B. For vec=1, 4B. OK.

Hmm wait, actually there's a subtlety: when stage is False, we don't allocate x_smem, and red_smem is first — 4B alignment fine.

Now double-check the sum reduce ordering hazard in staged path between pass1's st_s (x_smem writes) and pass2's ld_s: cta_max contains bar.sync in the middle (after leader stores). Pass-1 smem writes happen before warp butterfly... order: pass1 loop (stores to x_smem interleaved with global loads), then warp_max (shuffles), then lane0 store red, then bar.sync. The bar.sync guarantees all threads' pass-1 x_smem stores complete before any thread proceeds to pass 2. Good.

Pass 2: each thread reads only its own x_smem locations — but wait, is that right? Thread t reads locations {it*BLOCK*vec + t*vec ... } for its own t. Yes, same mapping as its writes. So even without the mid-reduce barrier it'd be safe, but we have it anyway.

Pass 2 writes e to x_smem (own locations), pass 3 reads own locations. Safe.

red_smem reuse between cta_max and cta_sum: cta_max ends with... currently my cta_max: warp_max, store, sync, read-reduce (no trailing sync). Then pass 2 (long). Then cta_sum: warp_sum (shuffles only), lane0 store red_smem[warp]. Hazard: thread A stores red_smem in cta_sum while thread B still reading red_smem in cta_max's tail. As analyzed, add trailing cta_sync() in cta_max. Similarly cta_sum needs no trailing sync before pass 3 (pass 3 doesn't touch red_smem; but pass-3 st_g to y global — no hazard). But at kernel end, no issue. I'll add trailing sync to both for safety/symmetry. Cost negligible.

Actually wait — also within cta_sum's tail, all threads read red_smem[0..3]; then trailing sync. Fine.

Now, potential issue: bar.sync with named barrier 0 and thread count 128 — example used that. Good.

One more consideration: `txl.ptx["max.f32"]` — is "max" a reserved word in their ptx namespace causing issues with bracket access? Bracket access is by string, no keyword issue. Good.

Also should I worry that ld.shared.f32 dest being local_scalar vs alloc_local element: example used ss[0] (alloc_local). mov used pk (local_scalar). I'll use local_scalar `m`, `s`, `red_tmp` as dests — should be OK.

Hmm, actually one risk: `txl.ptx["ld.shared.f32"](m, red_smem.ptr_to([w]))` where m is local_scalar — the DPS destination handling might expect an lvalue handle; local_scalar is an lvalue. Fine.

Now also: `txl.assign(m, txl.reinterpret(txl.uint32(0xFF7FFFFF), ty="float32"))` — txl.uint32(0xFF7FFFFF): 0xFF7FFFFF = 4286578687, fits uint32. Good. Alternatively use txl.float32(-3.4028234663852886e38) directly. Both fine; reinterpret is exact.

Actually simpler and less risky: `txl.float32(-3.402823466e38)`. That's representable (it's -FLT_MAX approx). Hmm, exact value -3.4028234663852886e38 rounds to... FLT_MAX = 3.4028234663852886e38 exactly representable? 2^128 - 2^104 ≈ 3.4028235e38; the decimal 3.4028234663852886e38 is exactly FLT_MAX's decimal repr, converts exactly. Use reinterpret anyway — it's precise and the README documents reinterpret with ty param. Actually README says `txl.reinterpret(value, ty=txl.uint32)`. String "float32" also used in example. OK.

Let me also handle: what if rows is 0? grid=rows=0 — CUDA disallows 0 grid? Actually grid dim 0 is invalid. But rows>=1 presumably. Not our concern.

Now let me also double check `txl.serial` yields loop var usable in arithmetic with Python ints: example `st = (ki * nthreads + tid) * vec`. Yes.

Type of `base`: int32. `base < n`: n Python int → constant. Good.

Now, about `txl.device_entry(launch=..., backend_config=...)` — required signature per the task. Note the task's skeleton passes backend_config as dict {"cuda": {"arch": "sm_80"}}. Keep.

Let me now also double check the vectorized shared load when stage and vec==4 but the address x_smem.ptr_to([base]) — base multiple of 4 → 16B aligned offset from a 16B-aligned base. Good.

One more potential concern: `txl.ptx[ST_S](x_smem.ptr_to([base]), xv[0], xv[1], xv[2], xv[3])` — st.shared.v4.f32 takes (ptr, v0..v3) with ptr first (store: sources after address). Example: `txl.ptx[ST_S_V4](x_smem.ptr_to([st]), xv[0], ...)` — yes ptr first for stores. Loads: dests first, ptr last. Good.

Global store: `txl.ptx[ST_G](y.ptr_to([r, base]), xv[0], ...)`.

Now write final code. Also add a tiny docstring. Only one code block with the module.

Also think: should grid be rows exactly? Yes, one CTA per row per requirements.

Also, maybe use `kernel_attrs=txl.cuda.KernelAttributes(...)`? Not needed.

Double-check helper definitions must be inside the kernel function (traced) since they emit device code — yes, define inside `softmax` after device_entry. But note: `txl.device_entry(...)` flat opens device region for remainder — helpers called within emit into device region. Good.

One catch: `ld_g`/`st_g` helpers close over `x`/`y`/`r` — fine.

Also the `if vec == 4:` inside helpers is Python-level (vec is Python int) — evaluated at trace. Good.

Let me write it.

```python
import math

import tirx_kernels.tirx_lite as txl

BLOCK = 128
NWARP = BLOCK // 32
FULL_MASK = 0xFFFFFFFF
LOG2E = 1.4426950408889634
NEG_FLT_MAX_BITS = 0xFF7FFFFF
SMEM_BUDGET = 48 * 1024 - 64  # leave room for the reduction scratch


def make_kernel(rows: int, n: int):
    vec = 4 if n % 4 == 0 else (2 if n % 2 == 0 else 1)
    nvec = n // vec
    iters = (nvec + BLOCK - 1) // BLOCK
    stage = (n * 4 + NWARP * 4) <= SMEM_BUDGET

    if vec == 4:
        LD_G, ST_G = "ld.global.v4.f32", "st.global.v4.f32"
        LD_S, ST_S = "ld.shared.v4.f32", "st.shared.v4.f32"
    elif vec == 2:
        LD_G, ST_G = "ld.global.v2.f32", "st.global.v2.f32"
        LD_S, ST_S = "ld.shared.v2.f32", "st.shared.v2.f32"
    else:
        LD_G, ST_G = "ld.global.f32", "st.global.f32"
        LD_S, ST_S = "ld.shared.f32", "st.shared.f32"

    @txl.kernel()
    def softmax(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
            backend_config={"cuda": {"arch": "sm_80"}},
        )

        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        if stage:
            x_smem = smem.alloc([n], "float32")  # first: keeps vector alignment
        red_smem = smem.alloc([NWARP], "float32")

        xv = txl.alloc_local([vec], "float32")
        m = txl.local_scalar("float32")   # row max
        s = txl.local_scalar("float32")   # row sum of exp
        red_tmp = txl.local_scalar("float32")

        def cta_sync():
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))

        def ld_g(ptr):
            if vec == 4:
                txl.ptx[LD_G](xv[0], xv[1], xv[2], xv[3], ptr)
            elif vec == 2:
                txl.ptx[LD_G](xv[0], xv[1], ptr)
            else:
                txl.ptx[LD_G](xv[0], ptr)

        def st_g(ptr):
            if vec == 4:
                txl.ptx[ST_G](ptr, xv[0], xv[1], xv[2], xv[3])
            elif vec == 2:
                txl.ptx[ST_G](ptr, xv[0], xv[1])
            else:
                txl.ptx[ST_G](ptr, xv[0])

        def ld_s(ptr):
            if vec == 4:
                txl.ptx[LD_S](xv[0], xv[1], xv[2], xv[3], ptr)
            elif vec == 2:
                txl.ptx[LD_S](xv[0], xv[1], ptr)
            else:
                txl.ptx[LD_S](xv[0], ptr)

        def st_s(ptr):
            if vec == 4:
                txl.ptx[ST_S](ptr, xv[0], xv[1], xv[2], xv[3])
            elif vec == 2:
                txl.ptx[ST_S](ptr, xv[0], xv[1])
            else:
                txl.ptx[ST_S](ptr, xv[0])

        def warp_max(v):
            peer = txl.local_scalar("uint32")
            for d in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(v, ty="uint32"),
                    txl.uint32(d),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.ptx["max.f32"](v, v, txl.reinterpret(peer, ty="float32"))

        def warp_sum(v):
            peer = txl.local_scalar("uint32")
            for d in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(v, ty="uint32"),
                    txl.uint32(d),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.assign(v, v + txl.reinterpret(peer, ty="float32"))

        def cta_max(v):
            warp_max(v)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), v)
            cta_sync()
            txl.ptx["ld.shared.f32"](v, red_smem.ptr_to([0]))
            for w in range(1, NWARP):
                txl.ptx["ld.shared.f32"](red_tmp, red_smem.ptr_to([w]))
                txl.ptx["max.f32"](v, v, red_tmp)
            cta_sync()

        def cta_sum(v):
            warp_sum(v)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), v)
            cta_sync()
            txl.ptx["ld.shared.f32"](v, red_smem.ptr_to([0]))
            for w in range(1, NWARP):
                txl.ptx["ld.shared.f32"](red_tmp, red_smem.ptr_to([w]))
                txl.assign(v, v + red_tmp)
            cta_sync()

        # ---- pass 1: row max (stage the row into shared memory when it fits)
        txl.assign(m, txl.reinterpret(txl.uint32(NEG_FLT_MAX_BITS), ty="float32"))
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                ld_g(x.ptr_to([r, base]))
                if stage:
                    st_s(x_smem.ptr_to([base]))
                for k in range(vec):
                    txl.ptx["max.f32"](m, m, xv[k])

        cta_max(m)

        # ---- pass 2: sum of exp(x - max); keep the exponentials around
        txl.assign(s, txl.float32(0.0))
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                if stage:
                    ld_s(x_smem.ptr_to([base]))
                else:
                    ld_g(x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["ex2.approx.f32"](xv[k], (xv[k] - m) * txl.float32(LOG2E))
                    txl.assign(s, s + xv[k])
                if stage:
                    st_s(x_smem.ptr_to([base]))

        cta_sum(s)
        txl.assign(s, txl.float32(1.0) / s)  # s becomes the reciprocal

        # ---- pass 3: normalize and store
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                if stage:
                    ld_s(x_smem.ptr_to([base]))
                else:
                    ld_g(x.ptr_to([r, base]))
                    for k in range(vec):
                        txl.ptx["ex2.approx.f32"](xv[k], (xv[k] - m) * txl.float32(LOG2E))
                for k in range(vec):
                    txl.assign(xv[k], xv[k] * s)
                st_g(y.ptr_to([r, base]))

    return softmax
```

Wait — math import unused now; remove or keep? "Only import tirx_kernels.tirx_lite (and math if needed)". I don't need math. Remove.

Issue: in pass 3 staged path, no recompute of exp — reads e from smem. Good.

Issue: pass 2 non-staged: ld_g then ex2. fine.

Issue: pass 2, when a thread loads and computes ex2 for its own elements — those threads with no elements contribute 0 to sum. Good.

Potential issue with `txl.reinterpret(v, ty="uint32")` where v is local_scalar and it's also the DPS destination in the same statement? No, separate statements.

One thing: `txl.ptx.shfl_sync.bfly.b32` — DPS dest `peer` local_scalar; example identical. Good.

Another: `warp` used as index red_smem.ptr_to([warp]) — warp expression int32. Good.

Concern: `txl.if_(lane == 0)` then_ with a store. Fine.

Concern: bar.sync count operand — must be multiple of warp? 128 fine.

Now, "s" reuse: after cta_sum, s holds total; assign reciprocal. Division `txl.float32(1.0) / s`. Should emit fdiv.rn? Possibly arith div → div.rn.f32, one per thread. fine.

Another check: pass 2 for the staged path stores e values with the same vector width — e.g., vec=4, st.shared.v4.f32 needs xv[0..3] all written — yes, all computed in loop.

Edge: NWARP=4, loop range(1,4) reads red_smem[1..3]. Good.

Potential issue: For stage=False, x_smem undefined in Python — referenced only under `if stage:` Python branches. `st_s`, `ld_s` helpers referenced only under stage branches. In pass 2 `if stage:` Python-level. Good.

One more: is `txl.alloc_local` correct spelling? Yes (README: txl.alloc_local(...) shorthand).

Is `txl.local_scalar("float32")` correct? Yes.

Check `txl.ptx.bar.sync` signature: example `txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(nthreads))`. Good.

Check comparison `base < n` where n is a Python int captured in closure — scale2 did exactly `col < n`. Good.

Check `txl.Tensor((rows, n), "float32")` annotations with Python ints — task skeleton. Good.

One subtle thing: the guard uses `base < n` but for vec>1, all vec elements valid. Confirmed since vec | n.

Also alignment for global v2/v4: x and y are separate tensors; both row-major with same n → same alignment properties. Torch base pointers 16B-aligned. OK.

What about `it * (BLOCK * vec)` overflow? int32, n up to huge — fine for realistic sizes.

Potential IR contract check issue: unused... all fine.

One thing I want to reconsider: `txl.ptx["ex2.approx.f32"]` — is attribute-style `txl.ptx.ex2.approx.f32` needed instead? Bracket indexing used in the official example with arbitrary instruction names (LD_G_V4 etc.), so bracket form is supported generally. Good.

Also `txl.ptx["max.f32"]` similar.

Also consider using fma for (x-m)*log2e — compiler will fuse. fine.

Hmm, one more: numerical stability — using max subtraction, ex2.approx. x~±100. exp(x-max) ∈ [~0, 1]. Sum ∈ [1, n]. y = e/s. Good.

What about NaN handling? Not required.

Potential concern about the trailing cta_sync in cta_max: needed as analyzed. Keep.

Also should pass 3 (staged) have a barrier issue: pass 2's st_s writes own locations, pass 3 ld_s reads own locations — same thread. But actually wait: is it exactly the same thread? base depends on tid; same tid → same bases. Yes.

But hmm — one more hazard: pass 2 stores to x_smem own slots; pass 3 reads own slots. Between them, cta_sum has bar.sync. Fine regardless.

Also, could the compiler reorder the smem store in pass 2 after... no, same address dependencies.

I think there might be an issue with `txl.serial(iters)` when iters could be 0? n>=1 → nvec>=1 → iters>=1. Good.

Now double check the DPS form of ld with 4 dests where some dests are alloc_local elements: example used iw[0..3] from alloc_local. Good; xv same.

st.global.v4.f32 with ptr first then 4 values: matches ST_G_V4 example (ptr first). Good.

One question: for vec==1, `txl.ptx[ST_G](ptr, xv[0])` — st.global.f32 [ptr], v. Good.

Final check of "one CTA (128 threads) per row" — grid=rows, block=128. ✓.

Also the interface: make_kernel returns softmax (the kernel object). ✓.

Let me also consider whether `stage` computation should account that smem_pool might have some overhead/alignment. Using 48KB-64 budget is safe.

Actually wait: is the static shared memory per-block limit 48KB, or does tirx handle dynamic smem with opt-in? smem_pool alloc likely becomes static shared memory in generated CUDA. 49152 static limit — ptxas errors above. Our budget respects that.

One more potential concern: when stage=True and vec==4, x_smem base must be 16-byte aligned. smem pool base is presumably aligned (usually to 16B at least, often 128B). If pool aligned to only 4 or 8 bytes, v4 shared access would misalign → error. Risk is low; pools typically align allocations (TVM smem pools align to at least 16 bytes, often to the vector width... hmm). To hedge, I could avoid vector shared ops and use scalar shared ops (4 separate ld.shared.f32). Shared memory bandwidth: scalar shared loads are fine; LSU throughput similar per instruction but 4x instructions. Given softmax is often memory-bound on global, scalar shared is acceptable and removes alignment risk. But the RMSNorm example used ld.shared.v4.f32 on a pool allocation directly, so pools support v4 — alignment is at least 16B there (they used x_smem.ptr_to([st]) with st multiple of 8 elements = 32B). So pool base is ≥16B aligned. Keep vector shared ops.

Actually the RMSNorm example allocated only x_smem and sum_sq_smem (4 bytes each? [bdy] floats). x_smem first. So can't infer alignment beyond that x_smem at offset 0 is v4-accessible → base aligned 16B. Our x_smem also first when stage. Good.

I'm fairly confident. One more look at potential issue: `txl.assign(xv[k], xv[k] * s)` — assign to alloc_local element shown in example (txl.assign(ss[0], ...)). Good.

Also `txl.assign(s, s + xv[k])` fine.

Also, in pass 1, `txl.ptx["max.f32"](m, m, xv[k])` — dest m, sources m and xv[k]. Good.

Consider: does the IR contract dislike `txl.float32(1.0) / s` division? Example used `/`. Fine.

Write final answer with brief comments.

## content
```python
# Row-wise softmax (numerically stable, sm_80): one CTA (128 threads) per row.
#
# Strategy (n is a Python constant at trace time, so the kernel is specialized):
#   * vector width vec = 4 / 2 / 1 chosen from n % 4 / n % 2, giving aligned
#     ld.global.v{4,2}.f32 / st.global.v{4,2}.f32 (rows start 16B/8B aligned);
#   * if the row fits in shared memory (n*4 + scratch <= 48KB) the row is staged
#     in smem during the max pass, the exponentials are rewritten into the same
#     smem slots during the sum pass, and the normalize pass reads them back —
#     exactly one global read and one global write per element;
#   * otherwise (very wide rows) the global row is re-read each pass;
#   * reductions: warp butterfly shuffles (max / add), warp leaders stage the
#     per-warp partials in shared memory, one bar.sync, every thread folds the
#     4 partials, one more bar.sync before the partials are reused;
#   * exp(v) = ex2.approx.f32(v * log2(e)) after subtracting the row max, so
#     arguments lie in [-inf, 0] and large |x| (~100) is handled stably.

import tirx_kernels.tirx_lite as txl

BLOCK = 128
NWARP = BLOCK // 32
FULL_MASK = 0xFFFFFFFF
LOG2E = 1.4426950408889634
NEG_FLT_MAX_BITS = 0xFF7FFFFF  # -FLT_MAX: identity for the max reduction
SMEM_BUDGET = 48 * 1024 - 64   # static smem budget incl. reduction scratch


def make_kernel(rows: int, n: int):
    vec = 4 if n % 4 == 0 else (2 if n % 2 == 0 else 1)
    nvec = n // vec  # exact: vec divides n
    iters = (nvec + BLOCK - 1) // BLOCK
    stage = (n * 4 + NWARP * 4) <= SMEM_BUDGET

    if vec == 4:
        LD_G, ST_G = "ld.global.v4.f32", "st.global.v4.f32"
        LD_S, ST_S = "ld.shared.v4.f32", "st.shared.v4.f32"
    elif vec == 2:
        LD_G, ST_G = "ld.global.v2.f32", "st.global.v2.f32"
        LD_S, ST_S = "ld.shared.v2.f32", "st.shared.v2.f32"
    else:
        LD_G, ST_G = "ld.global.f32", "st.global.f32"
        LD_S, ST_S = "ld.shared.f32", "st.shared.f32"

    @txl.kernel()
    def softmax(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
            backend_config={"cuda": {"arch": "sm_80"}},
        )

        r = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5

        smem = txl.smem_pool()
        if stage:
            # allocated first so the pool offset keeps the vector alignment
            x_smem = smem.alloc([n], "float32")
        red_smem = smem.alloc([NWARP], "float32")

        xv = txl.alloc_local([vec], "float32")  # vector staging registers
        m = txl.local_scalar("float32")  # row max
        s = txl.local_scalar("float32")  # row sum of exp, later its reciprocal
        red_tmp = txl.local_scalar("float32")

        def cta_sync():
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))

        def ld_g(ptr):
            if vec == 4:
                txl.ptx[LD_G](xv[0], xv[1], xv[2], xv[3], ptr)
            elif vec == 2:
                txl.ptx[LD_G](xv[0], xv[1], ptr)
            else:
                txl.ptx[LD_G](xv[0], ptr)

        def st_g(ptr):
            if vec == 4:
                txl.ptx[ST_G](ptr, xv[0], xv[1], xv[2], xv[3])
            elif vec == 2:
                txl.ptx[ST_G](ptr, xv[0], xv[1])
            else:
                txl.ptx[ST_G](ptr, xv[0])

        def ld_s(ptr):
            if vec == 4:
                txl.ptx[LD_S](xv[0], xv[1], xv[2], xv[3], ptr)
            elif vec == 2:
                txl.ptx[LD_S](xv[0], xv[1], ptr)
            else:
                txl.ptx[LD_S](xv[0], ptr)

        def st_s(ptr):
            if vec == 4:
                txl.ptx[ST_S](ptr, xv[0], xv[1], xv[2], xv[3])
            elif vec == 2:
                txl.ptx[ST_S](ptr, xv[0], xv[1])
            else:
                txl.ptx[ST_S](ptr, xv[0])

        def warp_max(v):
            peer = txl.local_scalar("uint32")
            for d in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(v, ty="uint32"),
                    txl.uint32(d),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.ptx["max.f32"](v, v, txl.reinterpret(peer, ty="float32"))

        def warp_sum(v):
            peer = txl.local_scalar("uint32")
            for d in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(v, ty="uint32"),
                    txl.uint32(d),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.assign(v, v + txl.reinterpret(peer, ty="float32"))

        def cta_max(v):
            warp_max(v)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), v)
            cta_sync()
            txl.ptx["ld.shared.f32"](v, red_smem.ptr_to([0]))
            for w in range(1, NWARP):
                txl.ptx["ld.shared.f32"](red_tmp, red_smem.ptr_to([w]))
                txl.ptx["max.f32"](v, v, red_tmp)
            cta_sync()  # red_smem is reused by the sum reduction below

        def cta_sum(v):
            warp_sum(v)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx["st.shared.f32"](red_smem.ptr_to([warp]), v)
            cta_sync()
            txl.ptx["ld.shared.f32"](v, red_smem.ptr_to([0]))
            for w in range(1, NWARP):
                txl.ptx["ld.shared.f32"](red_tmp, red_smem.ptr_to([w]))
                txl.assign(v, v + red_tmp)
            cta_sync()

        # ---- pass 1: row max (and stage the row into shared memory) ----
        txl.assign(m, txl.reinterpret(txl.uint32(NEG_FLT_MAX_BITS), ty="float32"))
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                ld_g(x.ptr_to([r, base]))
                if stage:
                    st_s(x_smem.ptr_to([base]))
                for k in range(vec):
                    txl.ptx["max.f32"](m, m, xv[k])

        cta_max(m)

        # ---- pass 2: sum of exp(x - max); keep the exponentials ----
        txl.assign(s, txl.float32(0.0))
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                if stage:
                    ld_s(x_smem.ptr_to([base]))
                else:
                    ld_g(x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["ex2.approx.f32"](xv[k], (xv[k] - m) * txl.float32(LOG2E))
                    txl.assign(s, s + xv[k])
                if stage:
                    # overwrite the staged row with its exponentials; each
                    # thread only touches the slots it read, so no hazard.
                    st_s(x_smem.ptr_to([base]))

        cta_sum(s)
        txl.assign(s, txl.float32(1.0) / s)  # s becomes 1 / sum

        # ---- pass 3: normalize and store ----
        with txl.serial(iters) as it:
            base = it * (BLOCK * vec) + tid * vec
            with txl.if_(base < n), txl.then_():
                if stage:
                    ld_s(x_smem.ptr_to([base]))
                else:
                    ld_g(x.ptr_to([r, base]))
                    for k in range(vec):
                        txl.ptx["ex2.approx.f32"](xv[k], (xv[k] - m) * txl.float32(LOG2E))
                for k in range(vec):
                    txl.assign(xv[k], xv[k] * s)
                st_g(y.ptr_to([r, base]))

    return softmax
```
