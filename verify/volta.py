"""Building block 2: run `volta compare` on a (reference, candidate) PTX pair and parse the verdict.

Input : VoltaCase = two PTX files + kernel names + launch dims + the memory/parameter setup
        (arrays with fixed base address / element width / length / kind, kernel params in order).
Output: VoltaResult with a normalized `status` plus details (mismatching elements, race location,
        timing, raw stdout/stderr, exact command).

Status values (normalized from VOLTA's text output, see third_party/volta/crates/volta_cli/src/main.rs
and crates/volta_analysis/src/eval/error.rs):
  EQUIV          every checked output element proved equal (over the reals)       exit 0
  NOT_EQUIV      >= 1 output element differs; `mismatches` + examples given       exit 1
  RACE           data race found (in `which` kernel)                               exit 1
  DEADLOCK       some threads blocked forever at a barrier                         exit 1
  OUT_OF_BOUNDS  memory access outside the declared arrays                         exit 1
  UNINIT         read of uninitialized register/memory reaching an output          exit 1
  FOOTPRINT      the two kernels wrote different sets of output elements           exit 1
  UNSUPPORTED    VOLTA cannot analyze this kernel/config (not structured-CTA, unsupported
                 instruction, lowering failure, instruction limit ...)            exit 1
  CONFIG_ERROR   our setup is wrong (unknown kernel name, bad array/param spec)    exit 1/2
  TIMEOUT        our wall-clock limit hit
  ERROR          anything not recognized (raw output kept)

Conditions / limits of what EQUIV means:
  * real-number semantics: float rounding, overflow, NaN/inf behaviour are NOT modeled;
  * only CTA 0 is executed, for exactly the given sizes/params;
  * exp must appear as `call __symexpf` (compile with verify.ptx mode="volta").
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import WORKSPACE

VOLTA_BIN = WORKSPACE / "third_party" / "volta" / "target" / "release" / "volta"
Z3_LIB_DIR = WORKSPACE / "tools" / "z3" / "z3" / "lib"


@dataclass(frozen=True)
class ArraySpec:
    """One global array visible to the kernels. kind: in | out | inout | index."""

    name: str
    base: int
    elem_bytes: int
    length: int
    kind: str

    def to_arg(self) -> str:
        if self.kind not in ("in", "out", "inout", "index"):
            raise ValueError(f"bad array kind {self.kind!r}")
        return f"{self.name}:{self.base:#x}:{self.elem_bytes}:{self.length}:{self.kind}"

    @property
    def end(self) -> int:
        return self.base + self.elem_bytes * self.length


@dataclass(frozen=True)
class VoltaCase:
    label: str
    ref_ptx: Path
    ref_kernel: str
    cand_ptx: Path
    cand_kernel: str
    ref_block: tuple[int, ...]
    cand_block: tuple[int, ...]
    arrays: tuple[ArraySpec, ...]
    params: tuple[str, ...]          # in declaration order: "ptr:<array>", "int:<v>", "float:<v>", "sym:<name>"
    check_arrays: tuple[str, ...]    # output arrays to compare
    grid: tuple[int, ...] = (1,)
    cand_grid: tuple[int, ...] | None = None
    dyn_shared: int = 0
    timeout_s: float = 600.0

    def validate(self) -> None:
        names = {a.name for a in self.arrays}
        spans = sorted((a.base, a.end, a.name) for a in self.arrays)
        for (b0, e0, n0), (b1, _, n1) in zip(spans, spans[1:]):
            if b1 < e0:
                raise ValueError(f"arrays {n0} and {n1} overlap")
        for p in self.params:
            if p.startswith("ptr:") and p[4:] not in names:
                raise ValueError(f"param {p} names an undeclared array")
        outs = {a.name for a in self.arrays if a.kind in ("out", "inout")}
        for c in self.check_arrays:
            if c not in outs:
                raise ValueError(f"check array {c} is not an out/inout array")
        if not self.check_arrays:
            raise ValueError("need at least one check array")


@dataclass
class VoltaResult:
    label: str
    status: str
    which: str | None = None              # "reference" | "candidate" | None
    detail: str = ""
    mismatches: int | None = None
    mismatch_examples: list[str] = field(default_factory=list)
    elements_checked: int | None = None
    instructions: int | None = None
    block_syncs: int | None = None
    exec_s: float | None = None
    vc_s: float | None = None
    wall_s: float | None = None
    exit_code: int | None = None
    cmd: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _dims(d: tuple[int, ...]) -> str:
    return ",".join(str(x) for x in d)


def build_command(case: VoltaCase, volta_bin: Path = VOLTA_BIN) -> list[str]:
    case.validate()
    cmd = [
        str(volta_bin), "--no-log-file", "compare", str(case.ref_ptx), str(case.cand_ptx),
        "--kernel1", case.ref_kernel, "--kernel2", case.cand_kernel,
        "-b", _dims(case.ref_block), "-g", _dims(case.grid), "--no-profile",
    ]
    if case.cand_block != case.ref_block:
        cmd += ["--block2", _dims(case.cand_block)]
    if case.cand_grid is not None and case.cand_grid != case.grid:
        cmd += ["--grid2", _dims(case.cand_grid)]
    for a in case.arrays:
        cmd += ["--array", a.to_arg()]
    for p in case.params:
        cmd += ["--param", p]
    for c in case.check_arrays:
        cmd += ["--check-array", c]
    if case.dyn_shared:
        cmd += ["--dyn-shared", str(case.dyn_shared)]
    return cmd


def _volta_env() -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{Z3_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return env


# ---- parsing (pure functions, unit-tested) ----------------------------------------------------

_EXEC_RE = re.compile(r"Exec: ([\d.]+)s\s+instructions: (\d+)\s+block syncs: (\d+)")
_VC_RE = re.compile(r"VC check: ([\d.]+)s .*elements: (\d+)")
_NOT_EQ_RE = re.compile(r"NOT EQUIVALENT: (\d+) mismatched element")
_ELEM_RE = re.compile(r"^\s+(\w+\[\d+\])\s*$", re.M)
_ROLE_RE = re.compile(r"^(?:error|Error): (reference|optimized) kernel: (.*)$", re.M)

_ERROR_PATTERNS = [  # (status, substring) checked in order
    ("RACE", "data race on"),
    ("DEADLOCK", "deadlock:"),
    ("OUT_OF_BOUNDS", "out-of-bounds"),
    ("FOOTPRINT", "written footprints differ"),
    ("UNINIT", "uninitialized"),
    ("UNSUPPORTED", "not a structured-CTA"),
    ("UNSUPPORTED", "unsupported"),
    ("UNSUPPORTED", "lowering failed"),
    ("UNSUPPORTED", "warp-op mismatch"),
    ("UNSUPPORTED", "instruction limit exceeded"),
    ("UNSUPPORTED", "value kind mismatch"),
    ("UNSUPPORTED", "trap reached"),
    ("CONFIG_ERROR", "configuration error"),
    ("CONFIG_ERROR", "no kernel named"),
    ("CONFIG_ERROR", "no kernel entry"),
]


def parse_output(label: str, exit_code: int, stdout: str, stderr: str) -> VoltaResult:
    r = VoltaResult(label=label, status="ERROR", exit_code=exit_code, stdout=stdout, stderr=stderr)
    if m := _EXEC_RE.search(stdout):
        r.exec_s, r.instructions, r.block_syncs = float(m[1]), int(m[2]), int(m[3])
    if m := _VC_RE.search(stdout):
        r.vc_s, r.elements_checked = float(m[1]), int(m[2])

    if exit_code == 0 and re.search(r"^EQUIVALENT$", stdout, re.M):
        r.status = "EQUIV"
        return r
    if m := _NOT_EQ_RE.search(stdout):
        r.status = "NOT_EQUIV"
        r.mismatches = int(m[1])
        r.mismatch_examples = _ELEM_RE.findall(stdout)
        r.detail = m[0]
        return r

    text = stderr + "\n" + stdout
    if m := _ROLE_RE.search(text):
        r.which = "reference" if m[1] == "reference" else "candidate"
        r.detail = m[2].strip()
    else:
        first = next((ln for ln in text.splitlines() if ln.strip().lower().startswith("error")), "")
        r.detail = first.split(":", 1)[-1].strip() if first else text.strip()[:500]
    low = r.detail.lower() or text.lower()
    for status, needle in _ERROR_PATTERNS:
        if needle.lower() in low:
            r.status = status
            break
    else:
        if exit_code == 2:          # clap argument error
            r.status = "CONFIG_ERROR"
    return r


# ---- runner ------------------------------------------------------------------------------------

def run_compare(case: VoltaCase, volta_bin: Path = VOLTA_BIN) -> VoltaResult:
    cmd = build_command(case, volta_bin)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=case.timeout_s, env=_volta_env())
    except subprocess.TimeoutExpired as e:
        return VoltaResult(label=case.label, status="TIMEOUT", detail=f"> {case.timeout_s}s",
                           wall_s=time.perf_counter() - t0, cmd=cmd,
                           stdout=(e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                           stderr=(e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))
    r = parse_output(case.label, proc.returncode, proc.stdout, proc.stderr)
    r.wall_s = time.perf_counter() - t0
    r.cmd = cmd
    return r
