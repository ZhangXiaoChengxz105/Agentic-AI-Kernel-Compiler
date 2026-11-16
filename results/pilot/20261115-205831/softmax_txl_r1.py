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
