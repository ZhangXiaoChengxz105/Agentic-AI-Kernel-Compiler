"""BUG (tiling): the tile count is k // 16 instead of ceil(k / 16), so the last k % 16 columns of A / rows of B
are never accumulated. Invisible when k is a multiple of 16.
Port of workloads/gemm/cuda/bugs/gemm_bug_ktail.cu.
Injected from gemm_txl_opt.py.

Original docstring: GEMM C = A @ B in tirx-lite for sm_80, OPTIMIZED:
classic 16 x 16 shared-memory tiling.
"""
import tirx_kernels.tirx_lite as txl

TILE = 16


def make_kernel(m: int, n: int, k: int):
    ntiles = k // TILE  # BUG: should be (k + TILE - 1) // TILE

    @txl.kernel()
    def gemm_txl(A: txl.Tensor((m, k), "float32"), B: txl.Tensor((k, n), "float32"),
                 C: txl.Tensor((m, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=((n + TILE - 1) // TILE, (m + TILE - 1) // TILE),
                                                      block=TILE * TILE),
                         backend_config={"cuda": {"arch": "sm_80"}})
        bx, by = txl.cta_id()
        tid = txl.thread_id()
        tx = tid % TILE
        ty = tid // TILE
        row = by * TILE + ty
        col = bx * TILE + tx
        pool = txl.smem_pool()
        As = pool.alloc([TILE, TILE], "float32")
        Bs = pool.alloc([TILE, TILE], "float32")
        a = txl.local_scalar("float32")
        b = txl.local_scalar("float32")
        acc = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.serial(ntiles) as t:
            ka = t * TILE + tx                 # column of A loaded by this thread
            kb = t * TILE + ty                 # row of B loaded by this thread
            txl.assign(a, txl.float32(0.0))
            with txl.if_(row < m), txl.then_():
                with txl.if_(ka < k), txl.then_():
                    txl.ptx.ld.global_.f32(a, A.ptr_to([row, ka]))
            txl.ptx.st.shared.f32(As.ptr_to([ty, tx]), a)
            txl.assign(b, txl.float32(0.0))
            with txl.if_(kb < k), txl.then_():
                with txl.if_(col < n), txl.then_():
                    txl.ptx.ld.global_.f32(b, B.ptr_to([kb, col]))
            txl.ptx.st.shared.f32(Bs.ptr_to([ty, tx]), b)
            txl.cuda.cta_sync()
            for kk in range(TILE):
                txl.ptx.ld.shared.f32(a, As.ptr_to([ty, kk]))
                txl.ptx.ld.shared.f32(b, Bs.ptr_to([kk, tx]))
                txl.assign(acc, acc + a * b)
            txl.cuda.cta_sync()
        with txl.if_(row < m), txl.then_():
            with txl.if_(col < n), txl.then_():
                txl.ptx.st.global_.f32(C.ptr_to([row, col]), acc)

    return gemm_txl
