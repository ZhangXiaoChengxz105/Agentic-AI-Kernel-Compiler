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
