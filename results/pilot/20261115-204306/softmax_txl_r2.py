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
