# tirx round 2

model=glm-5.3 in=10108 out=8790 cost=0.05070 latency=40.7s finish=stop effort=high usage={"completion_tokens": 8790, "completion_tokens_details": null, "prompt_tokens": 10108, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 7680, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 7212, "total_tokens": 18898}

## reasoning
The error: `txl.local_scalar` inside an `if_` frame introduces a new frame ("let") between if_ and else_. Actually the issue: inside `with txl.if_(warp == 0), txl.then_():` I declared `w = txl.local_scalar(...)` — a local declaration likely opens a "let" frame — and then `with txl.if_(lane < NWARP), txl.then_(): ... with txl.else_():` is nested inside that... wait the error says else_ occurred within the conditional (thread_id>>5==0), but other frames introduced since the If frame. The inner if_ is fine — else_ belongs to the inner if_? The error says "T.else_ must be called at the top of a T.if_()". So the else_ in question matched an If frame that is not the innermost. Hmm — perhaps the inner `with txl.if_(lane < NWARP), txl.then_():` combined style: `with txl.if_(cond), txl.then_():` — from the README, `with txl.if_(condition): with txl.then_():` or `with txl.if_(condition), txl.then_():`. Both shown in examples. The RMSNorm example uses nested withs. The small example uses `with txl.if_(col < n), txl.then_():`.

The problem is likely `w = txl.local_scalar("float32")` — local_scalar might be fine as statement... Actually the error mentions "other frames (e.g. if/else/let) had been introduced since the If frame". The `else_` call matches to the nearest enclosing If frame. If between the inner if_ then_ and the else_ another frame was introduced... In my code:

```
with txl.if_(warp == 0), txl.then_():
    w = txl.local_scalar("float32")
    with txl.if_(lane < NWARP), txl.then_():
        txl.ptx.ld.shared.f32(w, red.ptr_to([lane]))
    with txl.else_():
        txl.assign(w, ident)
```

Hmm, wait: `with txl.if_(cond), txl.then_():` — perhaps this form enters if_ frame and then_ frame, and on exit of the with, the then_ frame is popped but the if_ frame remains open until else_ or block end? Actually in the RMSNorm example they use fully nested:

```
with txl.if_(warp == 0):
    with txl.then_():
        with txl.if_(lane < bdy):
            with txl.then_():
                ...
            with txl.else_():
                ...
```

So the inner if with else is written nested. But the small verified example uses `with txl.if_(col < n), txl.then_():` without else. Maybe the tuple form `txl.if_(c), txl.then_()` — entering both contexts — when using else_, perhaps the tuple form works too? The error message indicates that when else_ was called, frames (let) had been introduced since the If frame — the `w = txl.local_scalar` likely introduces a let frame inside the then_ of the outer if, and then... hmm, but the else_ should match the inner if frame, not the outer.

Wait — maybe the issue: `with txl.if_(lane < NWARP), txl.then_():` — this combined form might enter the if_ frame and then_ frame as one `with` statement; when it exits, both frames close... no, then else_ would have no If frame to attach to, and the error would say no enclosing if. But the error says "While T.else_ did occur within the conditional based on (thread_id >> 5 == 0)" — that's warp == 0, the outer if. So the else_ matched to the outer if's frame! That means the inner if_ frame was NOT recognized... Hmm, or the else_ resolved to the outer If frame because the inner if_ frame was already closed? The error: "other frames (e.g. if/else/let) had been introduced since the If frame" — so between the outer If frame and the else_ call, other frames exist, and else_ tried to attach to the outer If frame. That suggests the else_ call's frame resolution walks outward and found the outer If frame, meaning the inner if_ frame... was matched and closed already? 

Actually maybe the semantics: `with txl.if_(c), txl.then_():` — on exit of the with body, the then_ frame closes AND the if_ statement completes (since with tuple context managers, both __exit__ are called). Then `with txl.else_():` — the else_ context manager looks for an open If frame; the inner if frame is closed, so it finds the outer if frame (still open, since we're inside its then_). But there are intervening frames (the let from local_scalar? or the inner then_?). Anyway, the fix: use the nested form exactly as the RMSNorm example does:

```
with txl.if_(cond):
    with txl.then_():
        ...
    with txl.else_():
        ...
```

Also, the `w = txl.local_scalar` inside if frame may introduce a "let" frame which is problematic between if_ and else_? In the RMSNorm example, no local_scalar declared between if_ and else_. To be safe, declare `w` outside the if, at function top with other locals.

Also note other issues to check:

1. `txl.local_scalar` as a statement inside branches — in original kernel I declared `cur`, `peer`, `out` inside the helper `cta_reduce` at what appears top-level of the helper (called from device body, so within serial loop frames). That worked? The trace failed at line 96 before reaching those. Better to declare all locals at top of kernel body or at least outside if frames.

2. `txl.Select` — used in `fmax` and `m` update. txl.Select(cond, a, b) — is that valid API? The README shows `txl.Select(choose_a, a_desc, b_desc)`. Yes, txl.Select exists. Good.

3. `cta_reduce` returns `out` local scalar declared inside — the function is a Python helper traced inline; declaring locals inside helper while inside `with txl.if_(warp == 0), txl.then_():` frame... The `out` local is declared after the if block, at serial-loop level. Fine, but to be safe move declarations up.

Actually simpler approach: avoid the if/else for lane<NWARP entirely. With BLOCK=128, NWARP=4, all lanes < 32 but only first 4 warps valid. Instead of if/else, load with clamped index: `idx = min(lane, NWARP-1)`? But that would double count... Alternative: pad red array to 32 entries? Shared memory zero-init isn't free. Alternative: have all 32 lanes participate, with lanes >= NWARP contributing ident by writing... Actually we can do: each lane reads red[lane] if lane < NWARP else ident — this requires the if/else. Alternatively use `txl.Select` for the value: load red at index `min(lane, NWARP-1)` and select ident when lane >= NWARP:

```
wv = txl.local_scalar("float32")
with txl.if_(warp == 0), txl.then_():
    with txl.if_(lane < NWARP), txl.then_():
        txl.ptx.ld.shared.f32(wv, red.ptr_to([lane]))
    with txl.else_():
        txl.assign(wv, ident)
```

That still uses else. Use nested form:

```
with txl.if_(warp == 0):
    with txl.then_():
        ...
        with txl.if_(lane < NWARP):
            with txl.then_():
                txl.ptx.ld.shared.f32(wv, red.ptr_to([lane]))
            with txl.else_():
                txl.assign(wv, ident)
```

That mirrors the RMSNorm example exactly. But wait — the RMSNorm example also declares no locals between if_ and else_. The w local there... In RMSNorm they reuse `ss[0]` (alloc_local array) instead of a fresh local_scalar inside the if. alloc_local might not introduce a "let" frame? Actually `txl.alloc_local` allocates a buffer, likely via let frames too. Hmm, in RMSNorm, `ss`, `rms`, etc. are all declared before the while loop. Inside the if/else they only use existing buffers. So the rule appears: don't declare anything between if_ and else_ frames.

So restructure: declare ALL locals (including wv, cur, peer, out) at the top of the kernel device body, before any control flow. Then cta_reduce uses them.

But cta_reduce is called twice with different locals... I can pass in the pre-declared locals. Let me restructure:

At top of kernel body (after device_entry, inside... note: helpers defined outside; cta_reduce takes locals as params):

```
cur = txl.local_scalar("float32")
peer = txl.local_scalar("uint32")
wv = txl.local_scalar("float32")
out = txl.local_scalar("float32")
```

Wait — but `local_scalar` returns an object; passing to Python helper is fine.

Hmm, but does `txl.local_scalar("float32")` at top-level of device body work? Yes, the scale2-style example... the RMSNorm example uses `txl.local_scalar` at device body top level (rms, row, goff). Good.

Then:

```
def cta_reduce(val, ident, combine):
    txl.assign(cur, val)
    # warp butterfly
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(cur, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.assign(cur, combine(cur, txl.reinterpret(peer, ty="float32")))
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
    with txl.if_(warp == 0):
        with txl.then_():
            with txl.if_(lane < NWARP):
                with txl.then_():
                    txl.ptx.ld.shared.f32(wv, red.ptr_to([lane]))
                with txl.else_():
                    txl.assign(wv, ident)
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(wv, ty="uint32"), ...)
                txl.assign(wv, combine(wv, txl.reinterpret(peer, ty="float32")))
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.f32(red.ptr_to([0]), wv)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
    txl.ptx.ld.shared.f32(out, red.ptr_to([0]))
    return out
```

But `return out` from a Python helper is fine — it returns the traced expression handle. However out is reassigned via txl.ptx.ld (DPS into local). Reading `out` after — in the RMSNorm style, they read `ss[0]` similarly. local_scalar: does reading it give the current value? `txl.assign(rms, ...)` and using rms in expressions — yes. For local_scalar written by ptx ld (DPS), reading it should work as in RMSNorm: `txl.ptx.ld.shared.f32(ss[0], ...)` — wait they use alloc_local array `ss` and index ss[0]. For local_scalar, `rms` is assigned via txl.assign. For ld into local_scalar: `txl.ptx.ld.shared.f32(out, ...)` — local_scalar should be passable as DPS dest. In my original I did that; can't verify but plausibly fine. To be extra safe, use `txl.alloc_local([1], "float32")` arrays like the RMSNorm example does for `ss`, and index [0]. The RMSNorm example demonstrates: `ss = txl.alloc_local([1], "float32")` then `txl.ptx.ld.shared.f32(ss[0], ...)` and uses `ss[0]` in expressions. That's a verified pattern. I'll use alloc_local([1]) for all DPS-written scalars: v, e, m, s, inv, cur, peer, wv, out.

Also `txl.reinterpret(acc, ty="uint32")` — RMSNorm uses `txl.reinterpret(acc[0], ty="uint32")`. With arrays fine. For local scalars, README says `txl.reinterpret(value, ty=txl.uint32)`. I'll use arrays consistently.

Also check: `combine(cur, ...)` where combine is Python lambda using txl.Select — `txl.Select(a > b, a, b)`. Is `>` on traced floats ok? The zero_matrix example uses `col < cols`. Yes.

Another concern: `txl.ptx.ex2.approx.f32(e, expr)` — DPS with computed expression operand. PTX call with expression should be fine. In RMSNorm, ptx calls take expressions like `ss[0] / txl.float32(hidden_size)`. Good.

Also `mtot` — I'll store cta_reduce result into m? cta_reduce returns `out[0]` expression; assign `txl.assign(mt[0], out[0])`? Or just use out[0] directly after each call. Since out is shared between calls, capture immediately: after first call, `txl.assign(m, out[0])` — but careful: the reduction for sum also writes out. Since we use it immediately after each call, fine. But to be clean, assign to dedicated locals:

```
txl.assign(m[0], out[0])  # after max reduction
...
txl.assign(s[0], out[0])  # after sum reduction
```

Actually simpler: have cta_reduce write result into a passed destination? PTX ld into `out` then caller assigns. Fine.

Another potential issue: the `with txl.if_(warp == 0), txl.then_():` tuple form — is it valid in general? The small verified example uses it (no else). The RMSNorm uses nested for the else-containing one. I'll use nested form wherever else_ is involved, and tuple form elsewhere (proven by example).

Also, `txl.local_scalar` between if and else was a problem; with alloc_local declared at top, no issue.

One more: in pass 2, `if USE_SMEM` Python-level branch — USE_SMEM is a Python bool, fine.

Also NTILE for n=200: TILE=512, NTILE=1. Fine.

Race check concern: cta_reduce called twice; second call writes red[warp] again — is there a hazard? After the first broadcast ld, I had a trailing bar.sync before returning so that all threads finish reading red[0] before warps overwrite red[warp] in the second call. Keep that.

Actually there's a subtle hazard within the second call: warp 0 writes red[0] then bar.sync, then all read red[0]. Meanwhile other warps already at bar.sync. Fine. Then final bar.sync before return protects the next call's `st red[warp]` vs this call's read of red[0]. Wait, but the ld of red[0] by all threads happens after bar.sync; the trailing bar.sync ensures all reads complete before any thread proceeds to the next call and writes red[warp]. Good.

Also shfl_sync within `with txl.if_(warp == 0), then_` — only warp 0 threads execute; FULL_MASK covers all 32 lanes of warp 0. Fine.

Also: `red.ptr_to([warp])` — warp is traced expression (tid>>5). ptr_to accepts expressions (RMSNorm uses ptr_to([warp])). Good.

Check ex2 for stability: v - mtot <= 0, times LOG2E <= 0, ex2 gives <= 1. Good. For x ~ ±100 N(0,1)*100: max subtracted, fine. exp underflow to 0, sum >= 1 (at least the max element gives 1). Good.

Edge: n small, e.g., n=1: NTILE=1, only thread 0 loads. Reductions handle. Good.

One concern: `txl.ptx.ex2.approx.f32` — does txl.ptx have ex2.approx.f32? PTX instruction `ex2.approx.f32`. Presumably txl.ptx exposes all PTX ops by name. Reasonable. Alternatively use txl.exp? Not documented. Keep ex2. Hmm, but risk: if ex2.approx.f32 isn't available, trace fails. Common softmax kernels use __expf which maps to ex2.approx. I'll keep it. Alternatively `txl.ptx.ex2.approx.ftz.f32`? Non-ftz safer for denormals (values like exp(-87)... after subtracting max, arguments are negative; exp(-60)~1e-26 fine in f32 normal range; ftz would flush results < 2^-126, negligible effect). Use ex2.approx.f32.

Also should double-check `txl.Select` signature: `txl.Select(cond, a, b)`. Used in README for descriptors. Should work for floats.

Another possible issue: declaring locals inside `with txl.serial(...)` body? All my alloc_local declarations are at device body top. Good.

Also `txl.assign(m[0], txl.Select(v[0] > m[0], v[0], m[0]))` — or use ptx max? Could use `txl.ptx.max.f32(res, a, b)`. PTX has max.f32. That might be cleaner and avoids Select with float comparison. Both fine; max.f32 is a real instruction. I'll use txl.ptx.max.f32. Hmm, risk of name availability; Select is documented. I'll use Select to be safe since it's in the README... Actually README documents Select only for descriptors but says "an FFI-backed expression constructor such as txl.Select accepts either wrapper" implying general expression constructor. I'll use Select.

Hmm wait, one more potential issue with Select for max with NaN? Inputs are finite; fine.

Also the combine functions passed as Python callables — traced fine.

Let me also double check the frame issue with `for delta in (...)` Python loop inside `with txl.if_(warp==0)/then_` — Python for loops don't create frames. Good.

One more: in cta_reduce I use `red.ptr_to` inside helper — red captured from closure. Fine.

Let me also reconsider: `txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))` — RMSNorm uses this form. Good. Could use txl.cuda variants but this matches.

Also potential Racecheck complaint: pass 2 writes xs[col] (shared) by the same thread that owns col, and pass 3 reads xs[col] by the same owning thread — no cross-thread sharing actually! Each thread reads back only slots it wrote. But warps may diverge... within a warp, threads execute pass 2 fully before pass 3 (same instruction stream, no barrier needed since same thread reads its own writes). Actually threads in a warp all execute pass 2 loop then pass 3 loop; shared memory writes by a thread are visible to itself. So no barrier needed between pass 2 and pass 3 for xs. But cta_reduce contains bar.syncs anyway. Fine.

But wait — Racecheck may still flag write-after-write / read between passes? No, same thread. OK.

Potential issue: in pass 1 with USE_SMEM, threads write xs; then cta_reduce's bar.sync ensures visibility (not needed for self reads but harmless).

Now write final module. Also keep UNROLL=4, TILE=512.

Double-check `ptr_to([r, col])` with r = cta_id (scalar for 1D grid). Example scale2 uses that. Good.

Also need to double check `txl.alloc_local([1], "float32")` then `txl.ptx.ld.global_.f32(v[0], ...)`. scale2 example used `v = txl.local_scalar("float32")` with `txl.ptx.ld.global_.f32(v, ...)`. So local_scalar works as DPS dest for ptx ld. RMSNorm used alloc_local arrays with ptx ld. Both patterns exist in verified examples. I'll use local_scalar for v, e, and alloc_local... hmm, to minimize risk, mix as proven: use local_scalar for scalars assigned by txl.assign and ptx (scale2 proves ptx ld into local_scalar works). Use alloc_local([1]) only where RMSNorm proves it. Actually simplest: use local_scalar everywhere; scale2 proves ld.global into local_scalar, RMSNorm proves ld.shared into ss[0] (alloc_local). Does ld.shared into local_scalar work? Both are just DPS into a local var handle; should be same mechanism. I'll use local_scalar throughout — it's the documented "writable scalars use txl.local_scalar".

But careful: reading local_scalar in expressions after ptx ld — scale2 doesn't read v after ld... it does: `v * txl.float32(2.0)`. Yes it reads. Good.

For reinterpret of local_scalar: `txl.reinterpret(cur, ty="uint32")` — README: `txl.reinterpret(value, ty=txl.uint32)`. RMSNorm uses string ty="uint32". OK.

Now, the warp butterfly in cta_reduce modifies `cur` which is passed as local. Since cta_reduce called twice, cur reused — fine, reassigned at start.

Write it:

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
        cur = txl.local_scalar("float32")
        wv = txl.local_scalar("float32")
        res = txl.local_scalar("float32")
        peer = txl.local_scalar("uint32")

        def warp_red(acc):
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(acc, ty="uint32"),
                    txl.uint32(delta),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.assign(acc, acc + txl.reinterpret(peer, ty="float32"))

        def cta_reduce(val, ident, is_max):
            txl.assign(cur, val)
            # warp-level butterfly
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(cur, ty="uint32"),
                    txl.uint32(delta),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                if is_max:
                    txl.assign(cur, txl.Select(cur > txl.reinterpret(peer, ty="float32"), cur, txl.reinterpret(peer, ty="float32")))
                else:
                    txl.assign(cur, cur + txl.reinterpret(peer, ty="float32"))
            ...
```

Hmm, cleaner: pass combine function:

```
def cta_reduce(val, ident, combine):
    txl.assign(cur, val)
    def step(): ...
```

But the butterfly writes into `cur` via combine. Let me just inline the warp reduction for cur with combine:

```
def warp_bfly(acc, combine):
    for delta in (16,8,4,2,1):
        txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret(acc, ty="uint32"), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
        txl.assign(acc, combine(acc, txl.reinterpret(peer, ty="float32")))
```

But acc must be a local_scalar to be assignable; `wv` and `cur` both are. Pass them.

combine lambdas: fmax via Select, fadd via +.

Full cta_reduce:

```
def cta_reduce(val, ident, combine):
    txl.assign(cur, val)
    warp_bfly(cur, combine)
    with txl.if_(lane == 0), txl.then_():
        txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
    with txl.if_(warp == 0):
        with txl.then_():
            with txl.if_(lane < NWARP):
                with txl.then_():
                    txl.ptx.ld.shared.f32(wv, red.ptr_to([lane]))
                with txl.else_():
                    txl.assign(wv, ident)
            warp_bfly(wv, combine)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.f32(red.ptr_to([0]), wv)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
    txl.ptx.ld.shared.f32(res, red.ptr_to([0]))
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
    return res
```

`return res` — returns the local scalar handle; caller uses it in expressions. Reading `res` after the helper returns: res is a traced local, expression reads refer to it. Should be OK. But careful: second cta_reduce call overwrites res. Caller must consume immediately. In code: after max reduction, `txl.assign(m, res)`? But m is used in pass 2 as `v - m`. Hmm — actually just use res directly where needed, or assign. I'll do:

```
mtot = cta_reduce(...)  # returns res
# use res directly in pass 2: (v - res)
```

But then second call: `stot = cta_reduce(...)` then use res for inv. Since passes are separated in code, using `res` directly is fine but confusing. Better: assign to m and s:

After first: `txl.assign(m, res)`. Hmm, but wait — actually cta_reduce's first op is `txl.assign(cur, val)` where val is `m` — reading m before overwriting, fine.

Hmm, one issue: `txl.assign(m, res)` where both local scalars — fine.

Wait, actually there's a subtlety with reading `res` in the caller after the helper — helpers using IRBuilder work (README says helpers work, "Emission and source locations also work for helpers using the native T namespace"). Fine.

Now pass structure with guarded ops. Also, `txl.Select(a > b, a, b)` — comparison `cur > peer_f`. OK.

Alternative to Select for max: use `txl.ptx.max.f32(dst, a, b)`. I recall txl.ptx exposes PTX ops generically. `max.f32` is standard PTX. Using it avoids Select-with-floats uncertainty. But DPS into a temp local... combine returns an expression, so Select fits the combine-returning-expression pattern. But actually I could restructure warp_bfly to not need combine to return an expression:

For max: `txl.ptx.max.f32(cur, cur, peer_f)` (DPS). For add: `txl.assign(cur, cur + peer_f)`.

So pass an op flag:

```
def warp_bfly(acc, is_max):
    for delta in (...):
        shfl...
        pf = txl.reinterpret(peer, ty="float32")
        if is_max:
            txl.ptx.max.f32(acc, acc, pf)
        else:
            txl.assign(acc, acc + pf)
```

But `pf = txl.reinterpret(...)` — assignment of Python var to expression is fine (no frame). This avoids Select entirely. But is `txl.ptx.max.f32` valid spelling? txl.ptx.st.global_.f32 shows dotted names with underscores for dots: `st.global_.f32` (global_ because 'global' is reserved). max.f32 → `txl.ptx.max.f32`. ex2.approx.f32 → `txl.ptx.ex2.approx.f32`. These should exist. Hmm, `max` is a Python builtin but attribute access fine.

I'll keep Select for max to minimize invented API? Select IS documented (README). Comparisons documented. I'll use Select. Actually, hmm: does Select evaluate both branches — no side effects here, fine.

Actually for the row max in pass 1, I can also use Select: `txl.assign(m, txl.Select(v > m, v, m))`.

Initial m = NEG_HUGE. If all values are -inf... not a concern.

Alternatively initialize m via first element? No, NEG_HUGE fine.

One more check: `txl.if_(lane < NWARP)` nested then/else — matching RMSNorm exactly:

```
with txl.if_(lane < bdy):
    with txl.then_():
        txl.ptx[LD_S_F32](ss[0], sum_sq_smem.ptr_to([lane]))
    with txl.else_():
        txl.assign(ss[0], txl.float32(0.0))
```

Yes. Note nothing declared between if_ and then_ and else_ — no let frames. My version complies (wv declared at top).

Also the outer: RMSNorm uses:
```
with txl.if_(warp == 0):
    with txl.then_():
        ...
```
Good.

Now full module. Also need to double check `txl.ptx.ex2.approx.f32(e, expr)` — DPS with expression operand. PTX calls in RMSNorm take expressions (`ss[0] / txl.float32(hidden_size) + txl.float32(eps)` passed to txl.rsqrt — that's txl.rsqrt, a builtin, not ptx). ptx calls with expression operands: `txl.ptx.st.global_.f32(..., v * txl.float32(2.0))` in scale2. Good, expressions allowed as sources.

Let me also double check `txl.float32(NEG_HUGE)` — txl.float32(value) constructs constant; NEG_HUGE = -3.4e38 fits float32 (max ~3.4028e38). Good.

Edge: rows dimension in Tensor annotation — `(rows, n)` static ints, like scale2. Good.

Also for USE_SMEM path with n up to ~12000: xs alloc n floats. Fine.

Actually, one more consideration: `red` smem alloc of NWARP floats — smem.alloc([NWARP], "float32"). Good.

Final check on the serial loop: `with txl.serial(NTILE) as it:` — NTILE is a Python int ≥ 1. Good. `base = it * TILE` — it is traced; TILE python int. `col = base + j*BLOCK + tid`. Compare `col < n` — n is Python int constant; fine.

Write final answer.

## content
```python
import tirx_kernels.tirx_lite as txl

FULL_MASK = 0xFFFFFFFF
NEG_HUGE = -3.4028234663852886e38  # -FLT_MAX: identity for the max reduction
LOG2E = 1.4426950408889634


def make_kernel(rows: int, n: int):
    BLOCK = 128
    NWARP = BLOCK // 32
    UNROLL = 4
    TILE = BLOCK * UNROLL
    NTILE = (n + TILE - 1) // TILE
    # Stage the row in shared memory when it fits in the default 48KB budget;
    # otherwise pass 2/3 re-read the row from global memory.
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

        # All writable scalars are declared up front: nothing (no let/if/else
        # frame) may be introduced between a T.if_() and its T.else_().
        v = txl.local_scalar("float32")
        e = txl.local_scalar("float32")
        m = txl.local_scalar("float32")
        s = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")
        cur = txl.local_scalar("float32")
        wv = txl.local_scalar("float32")
        res = txl.local_scalar("float32")
        peer = txl.local_scalar("uint32")

        def fmax(a, b):
            return txl.Select(a > b, a, b)

        def fadd(a, b):
            return a + b

        def warp_bfly(acc, combine):
            # Butterfly reduction of `acc` over the 32 lanes of this warp.
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(acc, ty="uint32"),
                    txl.uint32(delta),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.assign(acc, combine(acc, txl.reinterpret(peer, ty="float32")))

        def cta_reduce(val, ident, combine):
            # Block-wide reduction: warp butterfly, warp 0 across the warp
            # partials, broadcast through shared memory.
            txl.assign(cur, val)
            warp_bfly(cur, combine)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx.st.shared.f32(red.ptr_to([warp]), cur)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            with txl.if_(warp == 0):
                with txl.then_():
                    with txl.if_(lane < NWARP):
                        with txl.then_():
                            txl.ptx.ld.shared.f32(wv, red.ptr_to([lane]))
                        with txl.else_():
                            txl.assign(wv, ident)
                    warp_bfly(wv, combine)
                    with txl.if_(lane == 0), txl.then_():
                        txl.ptx.st.shared.f32(red.ptr_to([0]), wv)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            txl.ptx.ld.shared.f32(res, red.ptr_to([0]))
            # Keep `red` safe for reuse by the next reduction.
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))
            return res

        # ---- pass 1: row max (and stage the row in smem when it fits) ----
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
        txl.assign(m, mtot)

        # ---- pass 2: sum of exp(x - max); keep the exp values for pass 3 ----
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
                    # exp(v - m) via ex2; the argument is always <= 0, so the
                    # result is in (0, 1] and never overflows.
                    txl.ptx.ex2.approx.f32(e, (v - m) * txl.float32(LOG2E))
                    if USE_SMEM:
                        # Each thread rewrites only the slot it owns.
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
                        txl.ptx.ex2.approx.f32(e, (v - m) * txl.float32(LOG2E))
                    txl.ptx.st.global_.f32(y.ptr_to([r, col]), e * inv)

    return softmax_txl
```
