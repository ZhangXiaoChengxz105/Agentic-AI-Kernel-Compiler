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
