"""BUG (indexing, off-by-one): the variance loop covers col = 0..n (the tirx form of CUDA's i <= N), adding
(x[r, n] - mean)^2, i.e. the first element of the NEXT row, to the variance (for every n).
Port of workloads/layernorm/cuda/bugs/layernorm_bug_offbyone.cu.
Injected from layernorm_txl_opt.py.

Original docstring: LayerNorm over the last dim in tirx-lite for sm_80, OPTIMIZED:
block-parallel two-pass with tree reductions.
"""
import tirx_kernels.tirx_lite as txl

BLOCK = 128
EPS = 1e-5


def make_kernel(rows: int, n: int):
    iters = (n + BLOCK - 1) // BLOCK

    @txl.kernel()
    def layernorm_txl(x: txl.Tensor((rows, n), "float32"), w: txl.Tensor((n,), "float32"),
                      b: txl.Tensor((n,), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        red = txl.smem_pool().alloc([BLOCK], "float32")
        v = txl.local_scalar("float32")
        wv = txl.local_scalar("float32")
        bv = txl.local_scalar("float32")
        a = txl.local_scalar("float32")
        c = txl.local_scalar("float32")

        def block_sum(partial):
            txl.ptx.st.shared.f32(red.ptr_to([tid]), partial)
            txl.cuda.cta_sync()
            for k in (64, 32, 16, 8, 4, 2, 1):
                with txl.if_(tid < k), txl.then_():
                    txl.ptx.ld.shared.f32(a, red.ptr_to([tid]))
                    txl.ptx.ld.shared.f32(c, red.ptr_to([tid + k]))
                    txl.ptx.st.shared.f32(red.ptr_to([tid]), a + c)
                txl.cuda.cta_sync()
            total = txl.local_scalar("float32")
            txl.ptx.ld.shared.f32(total, red.ptr_to([0]))
            txl.cuda.cta_sync()            # everyone has read red[0] before red is reused
            return total

        # 1) mean
        s = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.serial(iters) as it:
            col = it * BLOCK + tid
            with txl.if_(col < n), txl.then_():
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.assign(s, s + v)
        mean = txl.local_scalar("float32")
        txl.assign(mean, block_sum(s) / txl.float32(n))

        # 2) variance (second pass over centered values)
        q = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.serial((n + BLOCK) // BLOCK) as it:   # BUG: n + 1 elements (i <= N)
            col = it * BLOCK + tid
            with txl.if_(col <= n), txl.then_():       # BUG: <= instead of <
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.assign(q, q + (v - mean) * (v - mean))
        rstd = txl.local_scalar("float32")
        txl.assign(rstd, txl.rsqrt(block_sum(q) / txl.float32(n) + txl.float32(EPS)))

        # 3) normalize + affine
        with txl.serial(iters) as it:
            col = it * BLOCK + tid
            with txl.if_(col < n), txl.then_():
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.ptx.ld.global_.f32(wv, w.ptr_to([col]))
                txl.ptx.ld.global_.f32(bv, b.ptr_to([col]))
                txl.ptx.st.global_.f32(y.ptr_to([r, col]), (v - mean) * rstd * wv + bv)

    return layernorm_txl
