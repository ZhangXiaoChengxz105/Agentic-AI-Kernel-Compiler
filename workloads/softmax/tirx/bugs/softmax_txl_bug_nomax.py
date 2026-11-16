"""BUG (numerical): exp is taken without subtracting the row max (overflows for large x).
Injected from softmax_txl_ref.py.

Original docstring: Row-wise softmax in tirx-lite for sm_80.

make_kernel(rows, n) -> txl.Kernel with x, y : float32 [rows, n]; one CTA (128 threads) per row.
Strided per-thread partial max / sum, shared-memory tree reductions with CTA barriers.
"""
import tirx_kernels.tirx_lite as txl

BLOCK = 128
NEG_BIG = -3.4e38          # stands in for -inf as the identity of max


def make_kernel(rows: int, n: int):
    iters = (n + BLOCK - 1) // BLOCK

    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        red = txl.smem_pool().alloc([BLOCK], "float32")
        v = txl.local_scalar("float32")
        a = txl.local_scalar("float32")
        b = txl.local_scalar("float32")

        def block_reduce(partial, combine):
            txl.ptx.st.shared.f32(red.ptr_to([tid]), partial)
            txl.cuda.cta_sync()
            for s in (64, 32, 16, 8, 4, 2, 1):
                with txl.if_(tid < s), txl.then_():
                    txl.ptx.ld.shared.f32(a, red.ptr_to([tid]))
                    txl.ptx.ld.shared.f32(b, red.ptr_to([tid + s]))
                    txl.ptx.st.shared.f32(red.ptr_to([tid]), combine(a, b))
                txl.cuda.cta_sync()
            out = txl.local_scalar("float32")
            txl.ptx.ld.shared.f32(out, red.ptr_to([0]))
            txl.cuda.cta_sync()            # everyone read red[0] before it is reused
            return out

        m = txl.local_scalar("float32", init=txl.float32(NEG_BIG))
        with txl.serial(iters) as it:
            col = it * BLOCK + tid
            with txl.if_(col < n), txl.then_():
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.assign(m, txl.max(m, v))
        mx = block_reduce(m, txl.max)

        s = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.serial(iters) as it:
            col = it * BLOCK + tid
            with txl.if_(col < n), txl.then_():
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.assign(s, s + txl.exp(v))
        total = block_reduce(s, lambda p, q: p + q)

        with txl.serial(iters) as it:
            col = it * BLOCK + tid
            with txl.if_(col < n), txl.then_():
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.ptx.st.global_.f32(y.ptr_to([r, col]), txl.exp(v) / total)

    return softmax_txl
