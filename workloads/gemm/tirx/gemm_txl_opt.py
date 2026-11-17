"""GEMM C = A @ B in tirx-lite for sm_80, OPTIMIZED: classic 16 x 16 shared-memory tiling.

make_kernel(m, n, k) -> txl.Kernel with A : float32 [m, k], B : float32 [k, n], C : float32 [m, n] (row-major).
Mirrors workloads/gemm/cuda/gemm_opt.cu. Launch: grid = (ceil(n / 16), ceil(m / 16)), block = 256 threads read
as 16 x 16 (tx = tid % 16, ty = tid // 16). Each CTA walks k in 16-wide steps: load a 16 x 16 tile of A and of
B into shared memory (zero-padding out-of-range elements), barrier, accumulate, barrier, next tile.
Works for any m, n, k >= 1. Plain SIMT f32 (no tensor cores, no cp.async).
"""
import tirx_kernels.tirx_lite as txl

TILE = 16


def make_kernel(m: int, n: int, k: int):
    ntiles = (k + TILE - 1) // TILE

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
