"""LayerNorm over the last dim in tirx-lite for sm_80 (hand-written reference; NOT shown to the LLM).

make_kernel(rows, n) -> txl.Kernel with x, y : float32 [rows, n] and w (gamma), b (beta) : float32 [n].
Mirrors workloads/layernorm/cuda/layernorm_ref.cu: one CTA per row, ONE thread (thread 0) does the whole row,
two passes: mean = sum(x) / n, var = sum((x - mean)^2) / n (biased, like torch),
y = (x - mean) * rsqrt(var + eps) * w + b.  eps = 1e-5 is a constant (tirx shapes and constants are static).
tirx-lite blocks have at least 32 threads, so the other 31 threads of the CTA do nothing.
One difference from the CUDA ref: the mean is summed shifted by the row's first element,
mean = x0 + sum(x - x0) / n (the same value over the reals). A plain serial float32 sum of a row with a large mean
(x ~ 1e4 + N(0, 1), the "offset" test) drifts by ~1e-2 and failed the LayerNorm tolerance on the A100 (job 24061076);
the shifted sum stays small.
"""
import tirx_kernels.tirx_lite as txl

BLOCK = 32                 # smallest tirx-lite block; only thread 0 works
EPS = 1e-5


def make_kernel(rows: int, n: int):
    @txl.kernel()
    def layernorm_txl(x: txl.Tensor((rows, n), "float32"), w: txl.Tensor((n,), "float32"),
                      b: txl.Tensor((n,), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=BLOCK),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        tid = txl.thread_id()
        v = txl.local_scalar("float32")
        wv = txl.local_scalar("float32")
        bv = txl.local_scalar("float32")
        s = txl.local_scalar("float32", init=txl.float32(0.0))
        q = txl.local_scalar("float32", init=txl.float32(0.0))
        x0 = txl.local_scalar("float32")
        mean = txl.local_scalar("float32")
        rstd = txl.local_scalar("float32")
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.ld.global_.f32(x0, x.ptr_to([r, 0]))
            with txl.serial(n) as i:                       # pass 1: mean = x0 + sum(x - x0) / n
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, i]))
                txl.assign(s, s + (v - x0))
            txl.assign(mean, x0 + s / txl.float32(n))
            with txl.serial(n) as i:                       # pass 2: variance of the centered values
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, i]))
                txl.assign(q, q + (v - mean) * (v - mean))
            txl.assign(rstd, txl.rsqrt(q / txl.float32(n) + txl.float32(EPS)))
            with txl.serial(n) as i:                       # normalize + affine
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, i]))
                txl.ptx.ld.global_.f32(wv, w.ptr_to([i]))
                txl.ptx.ld.global_.f32(bv, b.ptr_to([i]))
                txl.ptx.st.global_.f32(y.ptr_to([r, i]), (v - mean) * rstd * wv + bv)

    return layernorm_txl
