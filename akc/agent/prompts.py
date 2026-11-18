"""Prompts for the agent loop: the system prompt and the task prompt for any workload, in tirx-lite.

The agent writes tirx-lite only (README §2 "CAKE substitute"); the hand-written CUDA kernels are legacy.
The task prompt is built from the workload manifest, so the loop is not tied to one operator:
    description, signature["tirx"], tirx.factory_args / tirx.tensors, launch block, shape contract (pilot.contract_text),
    the tolerance of the OpSpec, the tirx-lite README and two examples (the same ones the pilot used),
    and a short list of pitfalls seen in the GLM pilot (notes/glm_pilot.md) or required by our validators.
Reference kernels are never shown to the model.

system_prompt() -> str
task_prompt(workload) -> str
extract_code(reply_text) -> str | None        the largest ```python block (pilot.extract_code)
"""
from __future__ import annotations

from ..task.reference import OPS
from ..task.workload import load_workload
from .pilot import TIRX_SYSTEM, TXL_DIR, contract_text
from .pilot import extract_code as _extract

SMALL_EXAMPLE = '''import tirx_kernels.tirx_lite as txl

def make_kernel(rows: int, n: int):
    @txl.kernel()
    def scale2(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={"cuda": {"arch": "sm_80"}})
        r = txl.cta_id()
        with txl.serial((n + 127) // 128) as it:
            col = it * 128 + txl.thread_id()
            with txl.if_(col < n), txl.then_():
                v = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.ptx.st.global_.f32(y.ptr_to([r, col]), v * txl.float32(2.0))
    return scale2'''

PITFALLS = """Pitfalls (seen in earlier runs, or required by our checks)
- `with txl.else_():` must sit directly inside `with txl.if_(cond):`, as a sibling right after the
  `with txl.then_():` block (never inside then_, never after other statements); follow the README's pattern.
- Target is sm_80 (Ampere): no Blackwell/Hopper-only features (tcgen05, TMA/TensorMap, wgmma, clusters, packed
  f32x2 instructions such as mul.rz.ftz.f32x2, mbarrier pipelines). Plain f32 scalar ops, shared memory, CTA barriers
  (txl.cuda.cta_sync()) and warp shuffles are fine. Every thread of the CTA must reach every barrier.
- Control flow (loop bounds, if conditions) may depend only on shapes, thread / CTA ids and constants, never on
  data values: use max/min instead of comparing data. Our formal verifier needs this.
- Do not use constants of magnitude 1.7e38 or more (e.g. -3.4e38 as the identity of a max); the verifier cannot
  represent them. Start a max reduction from the first element you load, or use a finite value such as -1e30.
- Only import tirx_kernels.tirx_lite (and math / typing / functools if needed). No torch, no I/O, no testing or
  benchmarking code: other imports and names such as os, open, eval are rejected before the module is loaded."""


def system_prompt() -> str:
    return TIRX_SYSTEM


def extract_code(text: str) -> str | None:
    return _extract(text or "", "tirx")


def task_prompt(workload: str) -> str:
    wl = load_workload(workload)
    if "tirx" not in wl.meta or "tirx" not in (wl.meta.get("signature") or {}):
        raise ValueError(f"workload {workload!r} has no tirx-lite contract (manifest 'tirx' and signature['tirx'])")
    op = OPS[workload]
    args = ", ".join(a.lower() for a in wl.meta["tirx"]["factory_args"])
    tensors = wl.meta["tirx"]["tensors"]
    readme = (TXL_DIR / "tirx_lite" / "README.md").read_text()
    example = (TXL_DIR / "norm" / "rmsnorm.py").read_text()
    return f"""Write a tirx-lite kernel for an NVIDIA A100 (sm_80).

Task: {wl.meta['description']}
Required interface: {wl.meta['signature']['tirx']}
The module must define make_kernel({args}) (static shapes: it is traced once per shape) returning a @txl.kernel
whose tensor parameters are, in this order: {', '.join(tensors)} (inputs first, the output last). The launch
configuration (grid, block, shared memory) is yours to choose inside txl.device_entry.
The shape names in the contract below are the manifest's ({', '.join(wl.shape_keys)}); make_kernel receives them
in that order.

{contract_text(wl, "tirx")}

Accuracy: float32 output compared with a float64 reference, |got - want| <= {op.atol:g} + {op.rtol:g} * |want|.
Inputs include large and small magnitudes and rows with one large outlier, not only N(0, 1).

{PITFALLS}

Aim for high performance on the benchmark shape.
Answer with exactly one ```python code block containing the complete module and nothing else.

=== tirx-lite authoring guide (README) ===
{readme}

=== Small verified sm_80 example (static shapes, global loads/stores, loops, guards) ===
```python
{SMALL_EXAMPLE}
```

=== Larger official example (RMSNorm; written for Blackwell, so ignore its f32x2 / sm_100-only parts and its
benchmark/test helpers, but its shared-memory, barrier and warp-shuffle idioms are valid on sm_80) ===
```python
{example}
```"""
