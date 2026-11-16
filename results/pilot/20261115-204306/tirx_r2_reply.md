# tirx round 2

model=glm-5.3 in=10707 out=11077 cost=0.06121 latency=51.3s finish=stop effort=high usage={"completion_tokens": 11077, "completion_tokens_details": null, "prompt_tokens": 10707, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 7680, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 8832, "total_tokens": 21784}

## reasoning
The issue: in `_block_reduce`, inside `with txl.if_(warp == 0), txl.then_():`, I then enter `with txl.if_(lane < WARPS), txl.then_():` and then `with txl.else_():` — the else_ must be entered while the if frame is top of stack. The error says the else_ occurred but other frames had been introduced since the If frame. Wait, actually looking at my code:

```python
with txl.if_(warp == 0), txl.then_():
    with txl.if_(lane < WARPS), txl.then_():
        txl.ptx["ld.shared.f32"](part, red.ptr_to([lane]))
    with txl.else_():
        txl.assign(part, identity)
```

The problem: `with txl.if_(lane < WARPS), txl.then_():` — the `then_()` block exits. But wait, when does then_ close? The then_ frame is exited after the with body. Then `txl.else_()` — but actually the issue is that when we call `txl.if_(cond)` and then `txl.then_()` as separate context managers in the same `with` statement... Hmm, in the RMSNorm example they nest them separately:

```python
with txl.if_(warp == 0):
    with txl.then_():
        ...
```

So the fix: don't combine `txl.if_(cond), txl.then_():` in one with — or rather, the issue may be that combining in one with statement causes the else_ frame confusion. Actually the guide shows `with txl.if_(condition): with txl.then_(): ... with txl.else_(): ...` where else_ is entered after exiting then_ block, both inside the if_ frame.

Actually wait — in my code, `with txl.if_(lane < WARPS), txl.then_():` — after this with exits, both then_ and if_ frames are... hmm, if if_ and then_ are entered together, exiting closes both? Then `txl.else_()` would be at top level of enclosing frame — no. The error says "While T.else_ did occur within the conditional based on (warp_id == 0), other frames (e.g. if/else/let) had been introduced since the If frame" — meaning the else_ was matched against the `warp == 0` if frame, but the `lane < WARPS` if frame was still... no wait, other frames had been introduced since the If frame — so the else_ attaches to the if frame based on warp==0, but the if frame (lane<WARPS) was pushed after it, and apparently when the combined `with A, B:` exits, only B is closed, A remains? No — with statement closes both on exit... unless `if_`'s `__exit__` defers closing until then_/else_ is used. Actually `txl.if_(cond)` likely returns a frame that stays open until then_ or else_ is used. In `with txl.if_(cond), txl.then_():` both enter; on exit, then_ closes, but the if_ frame... the guide example: `with txl.if_(condition): with txl.then_(): ... with txl.else_(): ...` — else_ entered after then_ exits, inside the if_ frame. So the if_ frame stays open until else_ is called or the if_ with-block exits.

In my code `with txl.if_(lane < WARPS), txl.then_():` — when this with exits, then_ exits then if_ exits? If if_ exits before else_ is called, then the else_ would attach to... the error suggests frames were introduced since the If frame (warp==0's frame? or lane<WARPS's?). The error mentions "the conditional based on (warp_id == 0)" — so else_ attached to the if frame for warp==0, and the frames introduced since include the lane<WARPS if frame. That means when `with txl.if_(lane < WARPS), txl.then_():` exited, the if_ frame did NOT close (maybe because combined with statement, if_ __exit__ doesn't close since then_ handled... no). Hmm, regardless: safest fix is to restructure to avoid else_, or use the nested form exactly as in the guide.

Simplest robust fix: avoid the combined `with txl.if_(c), txl.then_():` with a following else_. Replace the lane<WARPS if/else with arithmetic or a conditional expression. E.g., instead of if/else for partial loads, just load from smem with a clamped/safe index — but that reads red[lane] for lane>=4 which is out of bounds of size-4 array. Better: pad red to 32 elements (128 bytes) and initialize unused entries? We'd need to write them. Alternative: use `txl.Select`:

```python
loaded = txl.local_scalar("float32")
with txl.if_(lane < WARPS):
    with txl.then_():
        txl.ptx["ld.shared.f32"](v, red.ptr_to([lane]))
part = txl.Select(lane < WARPS, v, identity)
```

But we need part writable for warp reduce... The warp_reduce does combine(val, val, peer) — needs mutable local. We can assign: `txl.assign(part, txl.Select(lane < WARPS, v, identity))`. But v must be defined on all paths — it's a local_scalar, fine (uninitialized when false, but Select picks identity... actually Select evaluates both but IR-wise it's a select, fine; uninit register is fine since not used).

Actually simpler: since red smem array — allocate red with 32 elements but only 4 written... uninitialized smem read gives garbage — but we select it out. Still a read of uninitialized memory — racecheck/numsim might flag? Reading uninitialized shared memory is not a race in racecheck. But garbage float could be NaN... Select discards it, fine.

Cleanest: pad red to 32, and have all lanes of warp 0 read red[lane], but only lanes 0-3 have meaningful values. To make it deterministic, initialize red[4..31] to identity before the block reduce? That costs a barrier. Alternative: keep if/else but use the correct nesting form.

Actually the guide form:

```python
with txl.if_(condition):
    with txl.then_():
        ...
    with txl.else_():
        ...
```

So the else_ is entered while still inside the if_ frame. In my code I used the combined form everywhere. The RMSNorm example uses both: `with txl.if_(warp == 0), txl.then_():` combined form without else. So combined form is fine when there's no else. The issue is only when else_ follows.

So fix: in `_block_reduce`, change:

```python
with txl.if_(lane < WARPS), txl.then_():
    ...
with txl.else_():
    ...
```

Wait — is that what I wrote? Yes. The problem: with the combined form, when the `with` block exits, both frames exit, so the else_ has no matching if frame nearby... Actually the error says it did occur within the warp==0 conditional but other frames introduced since. Hmm, whichever — fix by using Select instead:

```python
v = txl.local_scalar("float32")
with txl.if_(lane < WARPS), txl.then_():
    txl.ptx["ld.shared.f32"](v, red.ptr_to([lane]))
part = txl.local_scalar("float32")
txl.assign(part, txl.Select(lane < WARPS, v, identity))
```

Hmm, but does local_scalar assign via txl.assign work with Select expression? Should. Alternatively use the properly-nested else form:

```python
with txl.if_(lane < WARPS):
    with txl.then_():
        txl.ptx["ld.shared.f32"](v, red.ptr_to([lane]))
    with txl.else_():
        txl.assign(v, identity)
```

But then v holds the value; warp_reduce mutates a scalar — I need part to be indexable? No, _warp_reduce operates on `val` scalar via closure, reassigning val. txl.assign(v, ...) — v is local_scalar, works. But warp_reduce does `combine(val, val, reinterpret(peer,...))` i.e., txl.ptx["add.f32"](val, val, b) — DPS with val as dest, val is a local_scalar expression — fine as in original code (ss[0] was alloc_local[1]; here local_scalar should be acceptable as ptx dest? In my code m, s, t are local_scalars used as ptx dests already... Actually in _fmax I do txl.ptx["max.f32"](dst, a, b) with dst=m a local_scalar. Does ptx instruction accept a local_scalar as destination? The RMSNorm example uses `pk = txl.local_scalar("uint64")` as dest of MUL_F32X2, and `rms` as... `txl.assign(rms, ...)`. And `v = txl.local_scalar("float32")` in the small example used with assign. pk used as ptx dest — yes local_scalar works as ptx dest. Good.

So the minimal fix: replace the problematic if/else with properly-nested form. But wait — was my combined form elsewhere also broken? The trace only failed at the else. The else_ error is specifically about else_. Also there's a subtle thing: my `with txl.if_(base < n), txl.then_():` combined form — worked in the small verified example, fine.

Also potential issue: `txl.Select` — but let's just use nested form per the guide.

Also the helper `_block_reduce` uses `txl.lane_id()` and `txl.warp_id()` — fine within device region (called from kernel body).

One more concern: in pass 2, when not use_smem, I store exp values to y then reload with LDG (coherent) — but `ld.global.v4.f32` after `st.global.v4.f32` by the same thread — is that safe without a barrier? Same thread wrote it, so it's coherent for that thread. Different threads don't read each other's elements. Fine. But actually I could avoid the reload: keep values in registers across the block reduce. With ntiles possibly >1, registers can't hold all tiles for large n. For the non-smem path n > 12284, so ntiles > ~96 — reload needed. Fine.

Actually, could optimize: if ntiles == 1, keep xf in registers and skip pass-3 reload. That's a trace-time specialization. Let me add it: if ntiles == 1, after computing sum, multiply xf in place and store, skipping third loop reload. And for smem path with ntiles==1 also could skip smem staging entirely. But keep it simple and correct — performance "aim high" but correctness first. I'll add the ntiles==1 register-resident fast path since it's easy: only one tile, values already in xf after pass 2. But block_reduce involves barriers — do registers survive? Yes, registers are per-thread and persist.

Wait but there's a subtlety: `_block_reduce` creates local scalars; fine.

Let me restructure: compute the reduction, then:

```python
if ntiles == 1:
    # xf still holds this thread's exp values (only threads with base < n loaded them,
    # but only those threads store)
    for k in range(vec):
        mul
    st_vec(STG, y.ptr_to([r, 0]))  # base = tid*vec, always < n since groups <= BLOCK and vec divides n
```

Wait if ntiles == 1, base = tid*vec < n for all tid < BLOCK since groups <= BLOCK. Since vec | n, base < n always. So no guard needed, and all 128 threads participate. Good.

Hmm, but careful: with ntiles==1, in pass 1 and 2 I guarded `base < n` — always true, harmless.

Actually let me also double-check the use_smem threshold: smem usage = WARPS*4 + n*4 bytes ≤ 48KB → n ≤ (49152-16)/4 = 12284. Good. But default smem without opt-in is 48KB per block on sm_80; static shared memory up to 48KB is fine. OK.

Another check: `_warp_reduce` uses `val` passed as local_scalar; shfl needs reinterpret to uint32 — `txl.reinterpret(val, ty="uint32")` where val is a local_scalar expression. In the RMSNorm example they reinterpret `acc[0]` (an alloc_local element). Should be OK for local_scalar too.

Also potential issue: `txl.lane_id()` and `txl.warp_id()` — in the RMSNorm example they used tid & 31 and tid >> 5. Both should be fine; guide says txl.warp_id and txl.lane_id exist. Keep them.

Also: `txl.ptx["ld.shared.f32"](v, red.ptr_to([lane]))` — v is local_scalar dest, fine.

Another concern from the trace failure: only one error point was reported, at _block_reduce. After fixing, other issues might surface, but I can only reason. Let me carefully review everything for potential trace issues:

1. `red = smem.alloc([WARPS], "float32")` — fine.
2. `xs = smem.alloc([n], "float32") if use_smem else None` — conditional Python expression evaluated at trace time — fine.
3. `txl.alloc_local([vec], "float32")` — fine.
4. `neg_inf = txl.reinterpret(txl.uint32(NEG_INF_BITS), ty="float32")` — guide says txl.reinterpret(value, ty=txl.uint32) — ty as string also used in RMSNorm (`ty="uint32"`). Fine.
5. `txl.float32(LOG2E)` — fine.
6. ptx instruction names via `txl.ptx["max.f32"]` — the RMSNorm example uses `txl.ptx[LD_G_V4]` string indexing. And attribute form `txl.ptx.st.global_.f32`. String form for max.f32 etc. should work.
7. `txl.ptx.shfl_sync.bfly.b32(...)` — used in RMSNorm.
8. `txl.ptx.bar.sync(...)` — used in RMSNorm as txl.ptx.bar.sync(uint32, uint32).
9. `txl.ptx["fma.rn.f32"](t, xf[k], log2e, shift)` — DPS form, fine.
10. `txl.ptx["ex2.approx.f32"](xf[k], t)` — fine.
11. `txl.assign(m, neg_inf)` — fine.

Potential numeric concern: ex2.approx.f32 accuracy — acceptable for softmax.

One more: in pass 2 for the non-smem path, I store to y and reload with LDG (coherent `ld.global.v...`). But the LDG name for vec>1 in non-smem pass 3: I used LDG defined as "ld.global.v%d.f32" — same-thread store-then-load is fine.

Hmm, also `ld.global.nc` on y? I used LDG_RO only for x. Good.

Also, for pass 3 in the smem path, the last `_block_reduce` for the sum ended with a final `_cta_sync()` — so smem writes in pass 2 (exp values staged) are visible to all threads. Good — racecheck should be happy.

Wait, but there's a race in the smem path: pass 2 reads xs (staged in pass 1) and writes xs (exp values). Pass 1's staging must be complete before pass 2 reads — the first `_block_reduce` (for max) ends with barriers: store to red by lane0 of each warp, sync, warp0 computes, store red[0], sync, all load red[0], sync. That last sync ensures pass-1 smem writes are visible. Good.

Racecheck within pass 2: each thread reads xs[base..] and writes xs[base..] — same locations, same thread. No cross-thread hazard. Good.

Now, the else_ fix. Let me use the nested form exactly per guide:

```python
with txl.if_(warp == 0):
    with txl.then_():
        with txl.if_(lane < WARPS):
            with txl.then_():
                txl.ptx["ld.shared.f32"](part, red.ptr_to([lane]))
            with txl.else_():
                txl.assign(part, identity)
        _warp_reduce(part, combine)
        with txl.if_(lane == 0):
            with txl.then_():
                txl.ptx["st.shared.f32"](red.ptr_to([0]), part)
```

The guide's canonical form. All my other combined `with txl.if_(c), txl.then_():` without else are fine per the verified example.

But hmm — in the failing error, it says else_ occurred within the conditional based on (warp_id == 0). That suggests the if_ frame it matched was warp==0's — because my `with txl.if_(lane < WARPS), txl.then_():` — when entered together, maybe the then_ frame is a child and on exit of the combined with... the else_ then attaches to warp==0's if frame, but the lane<WARPS if frame is still open ("other frames introduced since"). Whatever — nested form fixes it. Actually to be extra safe, use nested form for the outer warp==0 too (as in RMSNorm? No, RMSNorm uses combined form without else and it works).

Let me also double check `txl.else_()` exists as callable: guide shows `with txl.else_():`. Yes.

Also `txl.if_(lane == 0), txl.then_()` combined — fine (used in RMSNorm).

Now also add ntiles==1 fast path? It adds complexity and risk. The store in pass 3 reload from smem/y. For ntiles==1 (n <= 512), the kernel is small anyway. I'll keep it simple but add the register-resident path since it's a clear win for common small n... Risk: none really, it's trace-time Python branching. Let me add:

```python
if ntiles == 1:
    # values remain in xf
    for k in range(vec):
        txl.ptx["mul.f32"](xf[k], xf[k], inv)
    st_vec(STG, y.ptr_to([r, tid * vec]))
else:
    with txl.serial(ntiles) as it: ...
```

But careful: when ntiles == 1, pass 2's guard `base < n` always true, so all threads loaded xf. base = tid*vec; max base = 127*vec; since groups = n/vec <= 128, base/vec = tid <= 127 < groups? If groups <= 128 then for tid < groups, base < n. For tid >= groups, base >= n — guard false, xf uninitialized! E.g., n=200, vec=4, groups=50; tid=50..127 have base >= n. So in the fast path I must guard: `with txl.if_(tid < groups), txl.then_():`. Or keep the serial loop form which guards base < n. Simpler: keep the general loop even for ntiles==1 — but then the reload from smem happens. Alternatively in the general loop I could track "last tile" registers... overcomplicating.

Actually cleaner: in the general pass-3 loop, for the smem path with ntiles==1, the load from smem is cheap. Just keep general code. But we could skip the smem staging entirely when ntiles == 1 and keep xf in registers per tile... For multi-tile, registers per tile would be needed. Simplest optimization for correctness-focused: keep general loop. Performance is fine — smem loads are fast.

Hmm, but "aim for high performance" — a modest specialization: when ntiles == 1, skip smem entirely (use_smem path irrelevant) and skip pass-3 loads. Let me implement carefully:

Define `register_path = (ntiles == 1)`. Then:

- Pass 1: load x, compute max; stage to smem only if use_smem and not register_path.
- Block reduce max.
- Pass 2: load from smem if use_smem and not register_path, else from x (global). Compute exp, accumulate sum; store exp to smem if use_smem and not register_path, store to y if not use_smem and not register_path... wait if register_path we keep in xf and don't store. But guard: threads with base >= n skip. Then after block reduce sum, for register path: guard `base < n` (i.e., tid < groups) then multiply xf and store to y.

Hmm wait, if register_path and use_smem: n <= 512, fits smem, but we skip smem. Fine.

If register_path and not use_smem: n <= 512 always fits smem actually (12284 limit), so not use_smem implies n > 12284 implies ntiles > 96, never register_path. So register_path implies use_smem region. Fine.

This adds branching complexity. Given a trace failure already occurred, maybe prioritize robustness. But the register fast path is straightforward Python-level branching. I'll include it, carefully.

Actually, wait — one more consideration about pass-2 store to y in the non-register, non-smem path: after storing exp to y in pass 2, pass 3 reloads. That's the big-n path, fine.

Let me also reconsider: should pass 2 for the smem multi-tile path reload from smem rather than global x? Yes I do that (LDS). Good.

Now write the final module.

Also need to double check: `txl.lane_id()` returns lane within... warp? Guide: "txl.cta_id, txl.warp_id, txl.lane_id, and txl.thread_id are the entry-owned coordinates." lane_id is lane within warp (0-31), warp_id 0..block/32-1. Yes.

One more check on `_warp_reduce` with local_scalar val: `txl.reinterpret(val, ty="uint32")` — RMSNorm used txl.reinterpret(acc[0], ty="uint32") where acc[0] is element of alloc_local. val is a local_scalar expression — reinterpret should accept any expression. OK.

Also `combine(val, val, ...)` — ptx dest val, source val — reading and writing same local scalar, fine.

In `_block_reduce`, `part` local_scalar, `_warp_reduce(part, combine)` — modifies part in place. Good.

Also `identity` argument — for max it's neg_inf (an expression), for sum txl.float32(0.0). Fine.

Edge: n=1: vec=1, groups=1, ntiles=1. Pass 1: tid 0 loads, others m=-inf. Reduce max → x[0]. Pass 2: tid 0 computes exp(0)=1, s=1. Sum=1. inv=1. Store y=x... exp(0)/1=1. Correct.

Large |x| ~ 100: max subtracted, stable. exp(x-max) ∈ (0,1]. Good.

Check for x = -inf everywhere? Not required.

Now potential trace concern: `txl.ptx["ld.global.nc.f32"]` — does .nc exist as a name? ld.global.nc is standard PTX. The string-indexed txl.ptx[...] presumably accepts any PTX instruction name. RMSNorm used "ld.global.v4.b32". I'll trust it. But risk: if ld.global.nc isn't supported... To be safer, could use plain "ld.global.f32" for x too (non-nc). The .nc gives read-only cache benefit. Hmm, risk vs reward. The ptx namespace likely generic. I'll keep .nc but... actually, let me reduce risk: use plain `ld.global` for everything except maybe keep nc. Actually ld.global.nc requires the data to be read-only for the kernel's lifetime — true for x. I'll keep it. Hmm, but if the eval harness trace fails again, that's bad. The previous failure was a frame-structure error, resolved. Instruction names: RMSNorm used ld.global.v4.b32 and ld.shared.v4.f32, st.shared.f32, ld.shared.f32, shfl_sync.bfly.b32, bar.sync, mul.rz.ftz.f32x2, cvt.rn.f16x2.f32, mov.b32. Generic ones like max.f32, add.f32, mul.f32, fma.rn.f32, ex2.approx.f32, ld.global.nc.v4.f32 — plausible. The ptx table is probably a general PTX ISA mapping. I'll go with it.

Actually, wait — one risk with `ld.global.nc` + racecheck/NumSim: simulation of .nc loads should be fine.

Another risk: `txl.ptx["st.global.v2.f32"]` etc. fine.

Alignment for v2/v4: ld.global.v2.f32 requires 8-byte alignment; v4 requires 16-byte. Row r starts at element r*n; n%vec==0 ensures element offset multiple of vec, so byte offset multiple of 4*vec = 16 (v4) or 8 (v2). Base pointer of torch tensor is 256B-aligned. Good.

Now, register fast path: implement.

```python
register_path = ntiles == 1
```

Pass 1:
```python
txl.assign(m, neg_inf)
with txl.serial(ntiles) as it:
    base = (it * BLOCK + tid) * vec
    with txl.if_(base < n), txl.then_():
        ld_vec(LDG_RO, x.ptr_to([r, base]))
        for k in range(vec):
            txl.ptx["max.f32"](m, m, xf[k])
        if use_smem and not register_path:
            st_vec(STS, xs.ptr_to([base]))
```

Wait — `it * BLOCK + tid` — it is loop var; fine.

Pass 2:
```python
txl.assign(s, 0.0)
with txl.serial(ntiles) as it:
    base = (it * BLOCK + tid) * vec
    with txl.if_(base < n), txl.then_():
        if use_smem and not register_path:
            ld_vec(LDS, xs.ptr_to([base]))
        else:
            ld_vec(LDG_RO, x.ptr_to([r, base]))
        for k in range(vec):
            fma, ex2, add
        if not register_path:
            if use_smem:
                st_vec(STS, xs.ptr_to([base]))
            else:
                st_vec(STG, y.ptr_to([r, base]))
```

Hmm, when register_path, the xf values persist across _block_reduce — registers survive barriers, yes.

Pass 3:
```python
tot = _block_reduce(s, red, _fadd, txl.float32(0.0))
inv = 1.0/tot

if register_path:
    with txl.if_(tid * vec < n), txl.then_():
        for k in range(vec):
            txl.ptx["mul.f32"](xf[k], xf[k], inv)
        st_vec(STG, y.ptr_to([r, tid * vec]))
else:
    with txl.serial(ntiles) as it:
        base = (it * BLOCK + tid) * vec
        with txl.if_(base < n), txl.then_():
            if use_smem:
                ld_vec(LDS, xs.ptr_to([base]))
            else:
                ld_vec(LDG, y.ptr_to([r, base]))
            for k in range(vec):
                mul
            st_vec(STG, y.ptr_to([r, base]))
```

Wait — pass-3 smem load in multi-tile smem path: pass 2 wrote exp values to xs, and _block_reduce's trailing barrier ensures visibility. Good.

But wait, racecheck concern in pass 2 smem path multi-tile: pass 1 wrote xs (x values), then block reduce (barrier), pass 2 reads xs then writes xs at same addresses (same thread). Racecheck might flag write-after-read by same thread — fine, same thread is not a race.

One subtle racecheck issue: pass 2 thread A reads xs[i] (written by thread A in pass 1) then writes xs[i]. All same-thread. OK.

For non-smem multi-tile: pass 2 stores exp to y globally (each thread its own elements), pass 3 loads same elements by same thread. No cross-thread. Racecheck on global might be checked too. Same-thread RAW fine.

Now `_block_reduce` final structure with fixed else:

```python
def _block_reduce(val, red, combine, identity):
    lane = txl.lane_id()
    warp = txl.warp_id()
    part = txl.local_scalar("float32")
    out = txl.local_scalar("float32")
    _warp_reduce(val, combine)
    with txl.if_(lane == 0), txl.then_():
        txl.ptx["st.shared.f32"](red.ptr_to([warp]), val)
    _cta_sync()
    with txl.if_(warp == 0), txl.then_():
        with txl.if_(lane < WARPS):
            with txl.then_():
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
```

Hmm — is the nested `with txl.if_(lane < WARPS): with txl.then_(): ... with txl.else_(): ...` the right pattern? Guide: 

```python
with txl.if_(condition):
    with txl.then_():
        ...
    with txl.else_():
        ...
```

Yes exactly this. Good.

Alternatively avoid else entirely with Select to minimize risk:

```python
with txl.if_(warp == 0), txl.then_():
    v = txl.local_scalar("float32")
    with txl.if_(lane < WARPS), txl.then_():
        txl.ptx["ld.shared.f32"](v, red.ptr_to([lane]))
    txl.assign(part, txl.Select(lane < WARPS, v, identity))
    _warp_reduce(part, combine)
    ...
```

txl.Select is documented. But v uninitialized in else branch — Select is a select instruction, both operands are registers; uninitialized register is OK semantically but might trigger NumSim warnings about uninitialized reads? Select reads both operands at IR level... could produce NaN propagation? No — select picks one; sim evaluates both operand values then picks. Uninit register in sim might be undefined value, but select discards. Could still print warnings. The nested if/else is cleaner and matches the guide. Go with nested if/else.

Also: `txl.local_scalar` inside a helper called twice (two _block_reduce calls) — each call creates fresh locals. Fine.

One more consideration: helper functions `_cta_sync`, `_warp_reduce`, `_block_reduce` are module-level and called during trace — they use txl.* which requires active device region — yes, called from kernel body. Fine.

`_fmax`/`_fadd` as module-level helpers — fine.

Also note: in `_warp_reduce`, `peer = txl.local_scalar("uint32")` created inside the loop-iterating helper — created once per call (Python for loop over deltas is unrolled). Fine.

Check `txl.ptx["max.f32"](dst, a, b)` — DPS with 3 operands. Fine.

Now the exp trick: exp(x - mx) = 2^((x-mx)*log2e) = ex2(x*log2e + (-mx*log2e)). I compute shift = mx * (-log2e), then t = fma(x, log2e, shift). Note: this computes x*log2e - mx*log2e rather than (x-mx)*log2e — slight rounding difference, fine for softmax.

Precision with ex2.approx: ~2 ulp, fine.

Sum: since exp values ≤ 1 and n up to large, sum could be large but f32 handles.

inv = 1.0/tot: for the register path... division per thread (all threads compute same). Could use rcp.approx? div.rn.f32 is more accurate; only one per row per thread. Use `txl.ptx["div.rn.f32"](inv, txl.float32(1.0), tot)`? Or txl expression `txl.float32(1.0) / tot` — expression-level div. I'll use assign with expression; the RMSNorm example used `ss[0] / txl.float32(hidden_size)` as an expression. Good: `txl.assign(inv, txl.float32(1.0) / tot)`.

Hmm, one thing: `txl.assign(m, neg_inf)` where neg_inf is a reinterpret expression — fine.

Also check: in pass 1, threads where base >= n keep m = -inf. Good for max reduce.

Potential issue: `it * BLOCK + tid` types — it is int32 loop var presumably, tid int32. base = (it*128 + tid)*vec — int32. `x.ptr_to([r, base])` — r is cta_id (int32?), base int32; tensor indices... The small example used `x.ptr_to([r, col])` with col = it*128+thread_id. Fine.

Guard `base < n`: n is a Python int at trace time (static shapes). The comparison `base < n` — base is an expression, n Python int — fine (small example did `col < n`).

`tid * vec < n` in register path — fine.

Now smem alloc when register_path but use_smem: I allocate xs but never use it if register_path — wasted smem reducing occupancy. Better: `stage_smem = use_smem and not register_path`. So:

```python
xs = smem.alloc([n], "float32") if (use_smem and not register_path) else None
```

And use `xs is not None` checks. Since register_path implies n ≤ 512, and multi-tile smem path covers 512 < n ≤ 12284.

Simplify condition: `multi_smem = use_smem and not register_path`. Actually register_path implies ntiles==1 implies n ≤ 512*... wait vec can be 1: groups = n, ntiles = ceil(n/128); ntiles==1 → n ≤ 128 for vec=1. Generally n ≤ 128*vec ≤ 512.

OK. Let me also handle: what if n ≤ 512 but not use_smem — impossible as noted.

Write final code. Also keep LDG_RO with .nc for x loads; for the y reload (non-smem multi-tile path) use coherent ld.global.

Also `txl.float32(0.0)` for s init.

Double-check `txl.warp_id()` usage — returns warp index in block. With block=128, warp ∈ [0,4). red has WARPS=4 elements. Good.

Also check bar.sync count operand: `txl.ptx.bar.sync(txl.uint32(0), txl.uint32(BLOCK))` — RMSNorm did exactly this with nthreads. Good.

Potential issue: `txl.ptx["st.shared.f32"](red.ptr_to([warp]), val)` — red is smem tensor, ptr_to gives ptr. Good.

Now, one thing about `txl.if_` combined form with a helper-defined `lane == 0` where lane = txl.lane_id() computed in helper — expression fine.

Also: in `_block_reduce`, after `txl.ptx["ld.shared.f32"](out, red.ptr_to([0]))` I added an extra `_cta_sync()` at the end. Why? In the original I said it separates final broadcast load from reuse of red scratch by the next reduction. The next _block_reduce call writes red[warp] after its own warp reduce, and before that write there's a barrier... sequence: reduce1: ... ld out; sync. reduce2: warp_sum (no smem), lane0 writes red[warp], sync. Between reduce1's load and reduce2's write there IS reduce1's final sync — but the write happens after that sync? The sync at end of reduce1 orders everything before it. The write in reduce2 occurs after reduce1 returns (program order per thread), but other threads could... Thread A still executing `ld out` while thread B already in reduce2 writing red? The final sync in reduce1: all threads must reach it before any proceeds. The ld out happens BEFORE the final sync in my code. So after sync, all loads done, then writes are safe. Yes the trailing sync is needed and correct. Good.

But actually — is the trailing sync needed for the smem-staged pass-2 data too? Pass 2 writes xs; block_reduce for sum has internal syncs: lane0 writes red, sync (all xs writes visible), warp0 reduce, writes red[0], sync, all load red[0], sync. The load red[0] is after a sync that's after all xs writes — visible. Then pass 3 reads xs. Fine even without trailing sync, but harmless.

Hmm wait, one more ordering check in _block_reduce for the max: pass-1 smem staging writes xs; then _warp_reduce (registers only); lane0 stores red; sync → xs writes visible to all. Good.

Racecheck: pass 2 writes xs[base] by thread T; pass 3 reads xs[base] by same thread T; between them there are syncs. Fine.

Now finalize. Also `import math`? Not needed anymore. Only txl import.

Also module-level docstring fine.

One more possible trace pitfall: defining `_warp_reduce` with `for delta in (...)` Python loop unrolling — fine.

`combine` callback `_fmax(dst,a,b)` uses txl.ptx["max.f32"] — fine.

Let me also double check that `txl.local_scalar("float32")` can be used as ptx destination directly (e.g., `txl.ptx["st.shared.f32"]`... no, st's dest is a pointer). For `txl.ptx["ld.shared.f32"](part, ptr)` — part as dest — RMSNorm used `txl.ptx[LD_S_F32](ss[0], ...)` with ss = alloc_local([1]). pk (local_scalar) was used as dest of MUL_F32X2. So local_scalar as ptx dest works. And `txl.ptx["max.f32"](m, m, xf[k])` with m local_scalar dest — analogous to pk. Should be fine.

Also `txl.assign(shift, ...)` where shift local_scalar — fine.

Edge case n where vec chosen: n%4==0 → vec=4. E.g., n=200: 200%4=0 → vec=4, groups=50, ntiles=1 → register_path. Good. n=256: vec=4, groups=64, ntiles=1. n=1000: vec=4, groups=250, ntiles=2, use_smem. n=1002: vec=2, groups=501, ntiles=4. n=1001: vec=1.

Also rows=2 with grid=rows — fine.

I think there's one more subtle thing: `xf = txl.alloc_local([vec], "float32")` — with vec=1, alloc_local([1]) and xf[0]. Fine.

Also in `ld_vec` for vec==4 the ptx call has 5 args. Fine.

Write it out.

## content
```python
# Row-wise softmax (sm_80): one 128-thread CTA per row.
#
# Trace-time schedule (static n):
#   * vec = 4 / 2 / 1: widest vector width whose alignment is guaranteed at
#     every row start (row r begins at element offset r*n, so vec must divide n).
#   * Single-tile rows (n <= 128*vec) keep exp(x - max) in registers between
#     the sum reduction and the final store (no restaging at all).
#   * Rows that fit in the 48 KiB default shared-memory budget are staged into
#     smem once and rescaled from smem (1 global read + 1 global write).
#   * Very long rows use a 3-pass global variant: exp(x - max) is parked in y
#     and rescaled in place.
#
# Numerical stability: the row max is subtracted first (max.f32 reduction) and
# exp is evaluated as ex2.approx.f32(x * log2e - max * log2e) via one fma, so
# inputs with |x| ~ 100 are safe.

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
    # Butterfly reduction of the f32 local scalar `val` over the 32 lanes.
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
        with txl.if_(lane < WARPS):
            with txl.then_():
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
    register_path = ntiles == 1  # the whole row strip fits in one tile
    stage_smem = (n <= SMEM_LIMIT_ELEMS) and not register_path
    use_smem = n <= SMEM_LIMIT_ELEMS

    if vec == 1:
        LDG_RO = "ld.global.nc.f32"  # read-only path for x
        LDG = "ld.global.f32"  # coherent (y was written by this kernel)
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
        xs = smem.alloc([n], "float32") if stage_smem else None

        xf = txl.alloc_local([vec], "float32")
        t = txl.local_scalar("float32")
        m = txl.local_scalar("float32")
        shift = txl.local_scalar("float32")
        inv = txl.local_scalar("float32")
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

        # ---- pass 1: row max (and stage x into smem when useful) ----
        txl.assign(m, neg_inf)
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                ld_vec(LDG_RO, x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["max.f32"](m, m, xf[k])
                if stage_smem:
                    st_vec(STS, xs.ptr_to([base]))

        mx = _block_reduce(m, red, _fmax, neg_inf)

        # exp(x - mx) == ex2(x * log2e + shift), shift = -mx * log2e.
        txl.ptx["mul.f32"](shift, mx, nlog2e)

        # ---- pass 2: e = exp(x - mx), accumulate the row sum ----
        s = txl.local_scalar("float32")
        txl.assign(s, txl.float32(0.0))
        with txl.serial(ntiles) as it:
            base = (it * BLOCK + tid) * vec
            with txl.if_(base < n), txl.then_():
                if stage_smem:
                    ld_vec(LDS, xs.ptr_to([base]))
                else:
                    ld_vec(LDG_RO, x.ptr_to([r, base]))
                for k in range(vec):
                    txl.ptx["fma.rn.f32"](t, xf[k], log2e, shift)
                    txl.ptx["ex2.approx.f32"](xf[k], t)
                    txl.ptx["add.f32"](s, s, xf[k])
                if not register_path:
                    if stage_smem:
                        st_vec(STS, xs.ptr_to([base]))
                    else:
                        st_vec(STG, y.ptr_to([r, base]))

        tot = _block_reduce(s, red, _fadd, txl.float32(0.0))
        txl.assign(inv, txl.float32(1.0) / tot)

        # ---- pass 3: scale by 1/sum and store ----
        if register_path:
            # exp values are still live in xf; threads past the row end
            # never loaded (and never store).
            with txl.if_(tid * vec < n), txl.then_():
                for k in range(vec):
                    txl.ptx["mul.f32"](xf[k], xf[k], inv)
                st_vec(STG, y.ptr_to([r, tid * vec]))
        else:
            with txl.serial(ntiles) as it:
                base = (it * BLOCK + tid) * vec
                with txl.if_(base < n), txl.then_():
                    if stage_smem:
                        ld_vec(LDS, xs.ptr_to([base]))
                    else:
                        ld_vec(LDG, y.ptr_to([r, base]))
                    for k in range(vec):
                        txl.ptx["mul.f32"](xf[k], xf[k], inv)
                    st_vec(STG, y.ptr_to([r, base]))

    return softmax_txl
```
