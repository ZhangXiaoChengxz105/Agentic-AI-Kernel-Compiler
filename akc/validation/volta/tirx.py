"""tirx-lite kernels for VOLTA: trace at one static shape -> TIRx's generated CUDA -> VOLTA rewrites -> PTX.

build_tirx_ptx(mod, workload, shape, out_dir, label, rewrites=REWRITES) -> TirxPtx
volta_rewrite(cuda_src, rewrites=REWRITES) -> (rewritten source, {rewrite: count})
parse_entry(cuda_src) -> (kernel name, [(kind, name)])      kind "ptr" (tensor name, "_ptr" stripped) or "scalar"

Input : a tirx-lite module (loaded with backend.tirx.load_module, i.e. safety-scanned) whose factory takes the
        manifest's `tirx.factory_args`; the shape to trace (the caller sets rows = layouts.ROWS).
Output: TirxPtx = PTX for VOLTA + what VOLTA needs to launch it: entry name, grid, block, dynamic shared bytes,
        the tensor names in kernel-parameter order (shapes are baked in per trace, so a traced tirx-lite kernel
        takes only tensor pointers), the rewrites applied, and the seconds spent.
Conditions: tvm_tirx + tirx_kernels importable (CPU is enough: Kernel.source() only generates code), nvcc on PATH.
        PTX is built like the CUDA path's VOLTA build (backend.ptx mode="volta": no -lineinfo, VOLTA cannot lower a
        label followed by a `.loc`).

Rewrites (applied to the generated CUDA, identically for candidate and reference; counts are recorded):
  exp         expf( / __expf(  ->  __symexpf(
              VOLTA's exp hook. Exact in VOLTA's real-number model; the CUDA path does the same through EXP().
              Without it nvcc expands expf into ex2.approx + range reduction, which VOLTA does not read as exp.
  ex2         inline-asm `ex2.approx[.ftz].f32 d, t` and exp2f(t)  ->  d = __symexpf(t / fl32(log2 e))
              2^t is modelled as e^(t / 1.44269502f), i.e. ln 2 is taken as 1/fl32(log2 e): the exponent is off by a
              relative 1.3e-8, below the float32 rounding of t itself. With it, the GLM idiom
              ex2((x - m) * LOG2E) is exactly exp(x - m) for VOLTA (the division cancels the same float constant);
              a missing or wrong LOG2E factor stays visible. VOLTA rejects ex2 otherwise ("Unsupported instruction: Ex2").
  huge_const  a finite float constant with |c| >= 2^127, e.g. -FLT_MAX or -3.4e38 used as the identity of a max
              ->  +-INFINITY. VOLTA keeps constants as exact rationals with i128 parts and stops with
              "rational coefficient overflowed i128" on these. Float literals and TIRx's bit reinterprets
              (`uint v = (uint)<bits>; ... *(float *)(&(v))`) are both rewritten. This one is NOT exact in general:
              it is exact when the constant is the identity of a max/min and every input lies above it (true for all
              our kernels), but would change e.g. x + 3.4e38. It only touches constants VOLTA cannot represent at all.
  bar_count   a call of a TIRx barrier helper (`asm("bar.sync %0, %1;")` with register operands, from
              txl.ptx.bar.sync(id, count)) with literal arguments  ->  `bar.sync <id>;` with an immediate id.
              Exact when count == the CTA's thread count (PTX: without a count, all threads of the CTA take part);
              other counts are left alone. VOLTA needs an immediate barrier id and has no partial-CTA barriers.
  select_max  (on the PTX) `setp.<lt|le|gt|ge>[u].f32 p, a, b; ... selp.f32 d, x, y, p;` with {x, y} = {a, b} and p used
              nowhere else  ->  `max.f32 d, a, b` / `min.f32 d, a, b`; likewise with a NaN guard
              (`setp.nan.f32 q, a, a; or.pred r, p, q; selp.f32 d, a, b, r`, txl.max's NaN-propagating spelling).
              Exact over the reals (VOLTA's model has no NaN); in float32 only NaN handling differs. VOLTA keeps a
              compare-select as an opaque term, so max(-inf, x) inside a select stays "non-finite constant -inf";
              as max.f32 it folds to x (needed for txl.Select(a > b, a, b) reductions with a -inf identity).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ...backend.ptx import CompileError, compile_to_ptx

REWRITES = ("exp", "ex2", "huge_const", "bar_count", "select_max")
SOURCE_REWRITES = ("exp", "ex2", "huge_const", "bar_count")
LOG2E_F32 = "0x1.715476p+0f"          # fl32(log2 e) = 1.44269502162933349609375 (bits 0x3FB8AA3B)
F32_HUGE = 2.0 ** 127                 # VOLTA: constants >= this overflow its i128 rationals

PRELUDE = f"""
// ---- added by akc.validation.volta.tirx for VOLTA (not part of the kernel) ----
extern "C" __device__ float __symexpf(float x);                         // VOLTA's symbolic exp hook
static __forceinline__ __device__ float __volta_ex2(float t) {{ return __symexpf(t / {LOG2E_F32}); }}
#define __VOLTA_INF (__int_as_float(0x7f800000))
// ---- end of VOLTA additions ----
"""

_EXP_RE = re.compile(r"(?<![\w.])(?:__)?expf\s*\(")
_EXP2_RE = re.compile(r"(?<![\w.])exp2f\s*\(")
_ASM_EX2_RE = re.compile(
    r'(?:asm|__asm__)\s+(?:volatile|__volatile__)?\s*\(\s*"ex2\.approx(?:\.ftz)?\.f32\s+%0\s*,\s*%1\s*;"\s*'
    r':\s*"=f"\s*\(\s*(\w+)\s*\)\s*:\s*"f"\s*\(\s*(\w+)\s*\)\s*\)\s*;')
_NUM_COMMENT_RE = re.compile(r"/\*\s*[-+]?[0-9.]+(?:[eE][-+]?\d+)?\s*\*/")   # TVM's "/*-3.400000e+38*/"
_HEX_HUGE_RE = re.compile(r"(?<![\w.])0[xX]1(?:\.[0-9a-fA-F]*)?[pP]\+?127[fF]?(?![\w.])")
_DEC_RE = re.compile(r"(?<![\w.])(\d+\.?\d*[eE]\+?(\d+))[fF]?(?![\w.])")
_UINT_DECL_RE = re.compile(r"\buint\s+(\w+)\s*=\s*\(uint\)\s*(\d+)\s*;")
_BAR_HELPER_RE = re.compile(
    r'void\s+(\w+)\s*\(\s*\w+\s+\w+\s*(,\s*\w+\s+\w+\s*)?\)\s*\{\s*(?:asm|__asm__)\s+(?:volatile|__volatile__)?\s*'
    r'\(\s*"(?:bar|barrier)(?:\.sync)?(?:\.aligned)?\s+%0\s*(?:,\s*%1\s*)?;"[^;]*;\s*\}')
_LIT = r"\(\s*(?:uint|uint32_t|unsigned int|int)\s*\)\s*(\d+)|(\d+)[uU]?"
_ENTRY_RE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+)\s*'
                       r'\(([^)]*)\)\s*\{')
_PARAM_RE = re.compile(r"^\s*(?:const\s+)?(\w+)\s*\*\s*(?:__restrict__\s*)?(\w+)\s*$")
_FIRST_DECL_RE = re.compile(r'^(?:__forceinline__|extern\s+"C"|__global__|__device__|static\s|template\b)', re.M)
_SMEM_RE = re.compile(r'"cuda\.smem_required":\s*(\d+)')


class TirxBuildError(RuntimeError):
    """Tracing / code generation / nvcc failed for one shape. `stage` is trace | codegen | entry | nvcc."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message


@dataclass(frozen=True)
class TirxPtx:
    label: str
    ptx: Path
    cuda: Path                       # the rewritten CUDA that was compiled (kept for inspection)
    kernel: str
    grid: tuple[int, ...]
    block: tuple[int, ...]
    dyn_smem: int
    params: tuple[str, ...]          # manifest tensor names (`tirx.tensors`) in kernel-parameter order
    rewrites: dict = field(default_factory=dict)
    seconds: float = 0.0


def _is_huge(v: float) -> bool:
    return F32_HUGE <= abs(v)


def _huge_uint_bits(bits: int) -> bool:
    return ((bits >> 23) & 0xFF) == 0xFE           # finite, |value| in [2^127, 2^128)


def _rewrite_bar_calls(src: str, nthreads: int | None) -> tuple[str, int]:
    n = 0
    for h in _BAR_HELPER_RE.finditer(src):
        name, counted = h.group(1), h.group(2) is not None
        args = rf"\s*(?:{_LIT})\s*,\s*(?:{_LIT})\s*" if counted else rf"\s*(?:{_LIT})\s*"
        call = re.compile(rf"(?<![\w.]){re.escape(name)}\s*\({args}\)\s*;")

        def sub(m: re.Match) -> str:
            nonlocal n
            g = [x for x in m.groups() if x is not None]
            bid, count = int(g[0]), (int(g[1]) if counted else None)
            if bid > 15 or (counted and count != nthreads):
                return m.group(0)
            n += 1
            return f'asm volatile("bar.sync {bid};" ::: "memory");'
        src = call.sub(sub, src)
    return src, n


def volta_rewrite(src: str, rewrites=SOURCE_REWRITES, nthreads: int | None = None) -> tuple[str, dict]:
    """The CUDA-source rewrites listed in the module docstring; returns (source, counts). Unknown names raise
    ValueError; select_max (a PTX rewrite) is ignored here. nthreads: the CTA's thread count (bar_count only
    rewrites counted barriers whose count equals it)."""
    bad = set(rewrites) - set(REWRITES)
    if bad:
        raise ValueError(f"unknown rewrites {sorted(bad)}; known: {REWRITES}")
    rewrites = [r for r in rewrites if r in SOURCE_REWRITES]
    counts = {r: 0 for r in rewrites}
    if "exp" in rewrites:
        src, counts["exp"] = _EXP_RE.subn("__symexpf(", src)
    if "ex2" in rewrites:
        src, n_asm = _ASM_EX2_RE.subn(r"\1 = __volta_ex2(\2);", src)
        src, n_fn = _EXP2_RE.subn("__volta_ex2(", src)
        counts["ex2"] = n_asm + n_fn
    if "huge_const" in rewrites:
        src = _NUM_COMMENT_RE.sub("", src)
        src, n_hex = _HEX_HUGE_RE.subn("__VOLTA_INF", src)
        n_dec = 0

        def dec(m: re.Match) -> str:
            nonlocal n_dec
            if int(m.group(2)) >= 38 and _is_huge(float(m.group(1))):
                n_dec += 1
                return "__VOLTA_INF"
            return m.group(0)
        src = _DEC_RE.sub(dec, src)
        n_bits = 0
        for m in list(_UINT_DECL_RE.finditer(src)):
            name, bits = m.group(1), int(m.group(2))
            if not _huge_uint_bits(bits & 0xFFFFFFFF):
                continue
            uses = len(re.findall(rf"\b{re.escape(name)}\b", src)) - 1
            as_float = len(re.findall(rf"\(\*\(float \*\)\(&\({re.escape(name)}\)\)\)", src))
            if uses and uses == as_float:                  # only ever read as a float: TIRx's reinterpret
                inf_bits = (bits & 0x80000000) | 0x7F800000
                src = src.replace(m.group(0), f"uint {name} = (uint){inf_bits};", 1)
                n_bits += 1
        counts["huge_const"] = n_hex + n_dec + n_bits
    if "bar_count" in rewrites:
        src, counts["bar_count"] = _rewrite_bar_calls(src, nthreads)
    if any(counts.values()):
        m = _FIRST_DECL_RE.search(src)
        at = m.start() if m else 0
        src = src[:at] + PRELUDE + "\n" + src[at:]
    return src, counts


_SETP_RE = re.compile(r"^\s*setp\.(lt|le|gt|ge)u?\.f32\s+(%p\d+)\s*,\s*(%f\d+|0f[0-9A-Fa-f]{8})\s*,\s*"
                      r"(%f\d+|0f[0-9A-Fa-f]{8})\s*;", re.M)
_NAN_RE = re.compile(r"^\s*setp\.nan\.f32\s+(%p\d+)\s*,\s*(%f\d+)\s*,\s*(%f\d+)\s*;", re.M)
_ORP_RE = re.compile(r"^\s*or\.pred\s+(%p\d+)\s*,\s*(%p\d+)\s*,\s*(%p\d+)\s*;", re.M)
_SELP_RE = re.compile(r"^(\s*)selp\.f32\s+(%f\d+)\s*,\s*(%f\d+|0f[0-9A-Fa-f]{8})\s*,\s*"
                      r"(%f\d+|0f[0-9A-Fa-f]{8})\s*,\s*(%p\d+)\s*;", re.M)


def ptx_select_to_minmax(ptx: str) -> tuple[str, int]:
    """The select_max rewrite (see module docstring) on PTX text; returns (ptx, number of selp replaced)."""
    def uses(reg: str) -> int:
        return len(re.findall(rf"{re.escape(reg)}(?!\d)", ptx))
    cmp = {m.group(2): (m.group(1), m.group(3), m.group(4)) for m in _SETP_RE.finditer(ptx)}
    nan = {m.group(1): (m.group(2), m.group(3)) for m in _NAN_RE.finditer(ptx)}
    for m in _ORP_RE.finditer(ptx):                     # r = p | q with q = isnan(a): over the reals r = p
        r, p, q = m.group(1), m.group(2), m.group(3)
        if q in cmp and p in nan:
            p, q = q, p
        if p in cmp and q in nan and nan[q][0] == nan[q][1] and uses(p) == 2 and uses(q) == 2:
            cmp[r] = cmp[p]
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        ind, d, x, y, p = m.groups()
        if p not in cmp or uses(p) != 2:
            return m.group(0)
        op, a, b = cmp[p]
        greater = op in ("gt", "ge")                    # p = a > b (or a < b)
        if (x, y) == (a, b):
            fn = "max" if greater else "min"
        elif (x, y) == (b, a):
            fn = "min" if greater else "max"
        else:
            return m.group(0)
        n += 1
        return f"{ind}{fn}.f32 \t{d}, {a}, {b};"
    ptx = _SELP_RE.sub(sub, ptx)
    return ptx, n


def parse_entry(src: str) -> tuple[str, list[tuple[str, str]]]:
    m = _ENTRY_RE.search(src)
    if not m:
        raise TirxBuildError("entry", 'no `extern "C" __global__` kernel definition in the generated CUDA')
    params = []
    for p in (x for x in m.group(2).split(",") if x.strip()):
        pm = _PARAM_RE.match(p)
        if pm:
            name = pm.group(2)
            params.append(("ptr", name[:-4] if name.endswith("_ptr") else name))
        else:
            params.append(("scalar", p.strip()))
    return m.group(1), params


def _dims(v) -> tuple[int, ...]:
    v = v if isinstance(v, (tuple, list)) else (v,)
    return tuple(int(x) for x in v)


def launch_info(kernel) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """(grid, block, dynamic shared bytes) of a traced txl.Kernel (kernel.session.launch + its smem pool)."""
    launch = kernel.session.launch
    smem = getattr(launch, "dynamic_smem_bytes", None)
    if smem is None:
        m = _SMEM_RE.search(kernel.func.script())
        if m:
            smem = int(m.group(1))
        elif kernel.session.pool is not None:
            smem = int(kernel.session.pool.bytes)
    return _dims(launch.grid), _dims(launch.block), int(smem or 0)


def kernel_tensor_params(kernel, src: str, tensors: list[str]) -> tuple[str, list[str]]:
    """(entry name, manifest tensor names in kernel-parameter order) for TIRx's generated CUDA `src` of a traced
    txl.Kernel. TIRx orders parameters by name, so each one is mapped through the traced function's parameter
    position to the manifest's `tirx.tensors`. Raises TirxBuildError("entry") for scalar parameters (only static-shape
    kernels that take tensor pointers are supported) or a parameter that matches no tensor.
    Shared by the VOLTA build and the compiler-checks gate (compute-sanitizer launch)."""
    name, params = parse_entry(src)
    scalars = [p for kind, p in params if kind != "ptr"]
    if scalars:
        raise TirxBuildError("entry", f"kernel {name} takes non-pointer parameters {scalars}; this path "
                                      "supports static-shape tirx-lite kernels (tensor pointers only)")
    tensors = list(tensors)
    fparams = [str(getattr(p, "name", p)) for p in kernel.func.params]
    names = []
    for _, p in params:                 # kernel parameter -> the factory function's parameter position -> tensor
        if p in fparams and len(fparams) == len(tensors):
            names.append(tensors[fparams.index(p)])
        elif p in tensors:
            names.append(p)
        else:
            raise TirxBuildError("entry", f"kernel parameter {p!r} matches none of the tensors {tensors}")
    return name, names


def build_tirx_ptx(mod, workload_meta: dict, shape: dict, out_dir: Path, label: str,
                   rewrites=REWRITES) -> TirxPtx:
    """Trace `mod` at `shape`, rewrite its CUDA for VOLTA and compile it to PTX (see module docstring)."""
    from ...backend.tirx import make_kernel
    t0 = time.perf_counter()
    try:
        kernel = make_kernel(mod, workload_meta, shape)
    except Exception as e:  # noqa: BLE001  (any failure of the agent's Python while tracing)
        raise TirxBuildError("trace", f"{type(e).__name__}: {str(e).strip()[:1500]}") from e
    try:
        src = kernel.source()
    except Exception as e:  # noqa: BLE001
        raise TirxBuildError("codegen", f"{type(e).__name__}: {str(e).strip()[:1500]}") from e
    name, names = kernel_tensor_params(kernel, src, workload_meta["tirx"]["tensors"])
    grid, block, smem = launch_info(kernel)
    nthreads = 1
    for b in block:
        nthreads *= b
    new_src, counts = volta_rewrite(src, rewrites, nthreads)
    tag = "_".join(f"{k}{v}" for k, v in shape.items())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cu = out_dir / f"{label}_{tag}.cu"
    cu.write_text(new_src)
    (out_dir / f"{label}_{tag}.orig.cu").write_text(src)
    try:
        b = compile_to_ptx(cu, out_dir, mode="volta")
    except CompileError as e:
        raise TirxBuildError("nvcc", e.stderr[-2000:]) from e
    if name not in b.kernels:
        raise TirxBuildError("entry", f"entry {name} not in the PTX (entries: {b.kernels})")
    if "select_max" in rewrites:
        text, counts["select_max"] = ptx_select_to_minmax(b.ptx.read_text())
        b.ptx.write_text(text)
    return TirxPtx(label, b.ptx, cu, name, grid, block, smem, tuple(names), counts, time.perf_counter() - t0)
