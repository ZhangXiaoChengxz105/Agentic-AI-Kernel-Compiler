"""Mutation benchmark for the change classifier (and, on a GPU, for configurations A/C).

Why: the 11 injected bugs and the classifier rules were written by the same person, so 11/11 says little about
how the classifier does on edits it was not written for. Here the edits are generated mechanically from fixed
mutation operators, on our kernels AND on kernels written by GLM-5.3, and each operator has a set of acceptable
categories (the ground truth).

  mutants(src, lang) -> [Mutant(op, line, desc, src, accept)]     deterministic, <= PER_OP sites per operator
  classify_mutant(workload, base_src, m, llm=None) -> categories  via changeaware.plan (rules, + LLM if given)

Operators (accept = categories that count as a correct classification):
  cmp     <-> <= / > <-> >= in a loop/guard condition                         indexing
  idx1    an array subscript gets +1                                           indexing
  const   an integer literal >= 2 in code is halved / incremented            indexing, tiling, memory, sync
  define  a #define tile/block constant is doubled                             tiling
  nosync  one barrier statement deleted                                        sync
  shfl    a warp shuffle changed (xor -> down)                                  sync
  shfl_off  the start offset of a shuffle loop halved (16 -> 8)               sync, indexing
  aop     a float arithmetic operator changed (+ -> -, * -> /, += -> =); index/pointer math excluded   numerical
  math    max <-> min                                                          numerical
  init    an initial value changed (-INFINITY / -FLT_MAX / 0.0f)             numerical
  negate  an if-condition negated                     [held out]              indexing
  smem    a shared-memory array size halved             [held out]              memory, tiling
  dtype   a float local made double / float16           [held out]              numerical
  rename  a local variable renamed everywhere (NO semantic change)            (none: any label is spurious)
  format  statements re-wrapped (NO semantic change)                           (none)

Mutant.bug = False marks edits that cannot change any output (an array declared 1 larger, float -> double): they
still test the classifier but are not counted for detection. With --detect, each base kernel is first run through
the fixed tests and the change-aware floor tests; a base that fails by itself is not used for detection (3 of the
GLM held-out drafts are wrong already), and when A already catches a mutant C is not re-run (C includes A's tests).

CLI:  akc mutants [--set bases|heldout|all] [--llm-classify [--llm-model M]] [--per-op 3]   (CPU; LLM spends coins)
      akc mutants --classifier ast|ir [--only LABELS] [--compare-with OTHER.json]
          ir (default): tirx-lite rows use the IR change analysis (changeaware.irdiff / classify_ir; CUDA rows keep
          the text classifier); ast: tirx-lite rows use the Python-AST classifier as before. --compare-with prints
          an agreement matrix (verdict here vs verdict there, matched by base / operator / line / edit).
      akc mutants --set all --per-op 1 --detect --device gpu                (GPU: is each mutant caught by A / C?)
      e.g. sbatch --time=00:55:00 scripts/slurm/gpu_job.sbatch akc mutants --set all --per-op 1 --detect --device gpu
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ..paths import RESULTS, ROOT

PER_OP = 3
STRUCTURAL = ("indexing", "tiling", "memory", "sync")

BASES = [  # (label, workload, path relative to the repo root)
    ("softmax_opt", "softmax", "workloads/softmax/cuda/softmax_opt.cu"),
    ("layernorm_opt", "layernorm", "workloads/layernorm/cuda/layernorm_opt.cu"),
    ("gemm_opt", "gemm", "workloads/gemm/cuda/gemm_opt.cu"),
    ("glm_A_r1", "softmax", "results/pilot/20261115-203841/softmax_glm_r1.cu"),
    ("glm_B1_r3", "softmax", "results/pilot/20261115-204303/softmax_glm_r3.cu"),
    ("glm_B2_r1", "softmax", "results/pilot/20261115-204348/softmax_glm_r1.cu"),
    ("glm_B3_r1", "softmax", "results/pilot/20261115-204412/softmax_glm_r1.cu"),
    ("softmax_txl_ref", "softmax", "workloads/softmax/tirx/softmax_txl_ref.py"),
]
# Held out: kernels NOT looked at while the classifier rules were revised on 2026-11-16 (the rules were tuned on
# BASES, so BASES numbers after that are training-set numbers; report HELDOUT as the honest estimate).
HELDOUT = [
    ("glm_A_r2", "softmax", "results/pilot/20261115-203841/softmax_glm_r2.cu"),
    ("glm_A_r3", "softmax", "results/pilot/20261115-203841/softmax_glm_r3.cu"),
    ("glm_B1_r1", "softmax", "results/pilot/20261115-204303/softmax_glm_r1.cu"),
    ("glm_B1_r2", "softmax", "results/pilot/20261115-204303/softmax_glm_r2.cu"),
    ("glm_txl_C_r2", "softmax", "results/pilot/20261115-203843/softmax_txl_r2.py"),
    ("glm_txl_D1_r2", "softmax", "results/pilot/20261115-204306/softmax_txl_r2.py"),
    ("glm_txl_D2_r1", "softmax", "results/pilot/20261115-205635/softmax_txl_r1.py"),
    ("glm_txl_D3_r1", "softmax", "results/pilot/20261115-205831/softmax_txl_r1.py"),
]


@dataclass
class Mutant:
    op: str
    line: int
    desc: str
    src: str = field(repr=False)
    accept: tuple[str, ...]
    bug: bool = True             # False: the edit cannot change any output (larger array, more precision);
                                 # still a classifier test, but not counted for detection


def _code_lines(src: str, lang: str) -> list[int]:
    """0-based indices of lines that hold code (not comments, preprocessor, docstrings, blank)."""
    out, in_block, in_doc = [], False, False
    for i, line in enumerate(src.splitlines()):
        t = line.strip()
        if lang == "cuda":
            if in_block:
                in_block = "*/" not in t
                continue
            if t.startswith("/*"):
                in_block = "*/" not in t
                continue
            if not t or t.startswith("//") or t.startswith("#"):
                continue
        else:
            if t.count('"""') == 1:
                in_doc = not in_doc
                continue
            if in_doc or not t or t.startswith("#") or t.startswith('"""'):
                continue
        out.append(i)
    return out


def _spread(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _sub_line(src: str, i: int, new_line: str) -> str:
    lines = src.splitlines()
    lines[i] = new_line
    return "\n".join(lines) + "\n"


def _code_part(line: str, lang: str) -> str:
    cut = line.find("//") if lang == "cuda" else line.find("#")
    return line if cut < 0 else line[:cut]


_CMP = re.compile(r"(?<![<>=!\-])(<=|>=|<|>)(?![<>=])")
_COND = {"cuda": re.compile(r"\b(for|if|while)\s*\("), "tirx": re.compile(r"txl\.if_\(|\bif\b|\bwhile\b|range\(")}
_SUBSCRIPT = {"cuda": re.compile(r"\b(\w+)\[([^\[\]]+)\]"), "tirx": re.compile(r"ptr_to\(\[([^\[\]]+)\]\)")}
_INT = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_DEFINE = re.compile(r"^(\s*#\s*define\s+\w*(?:TILE|BLOCK|BM|BN|BK|TPB|THREADS|WARPS)\w*\s+)(\d+)")
_SYNC = {"cuda": re.compile(r"^\s*__syncthreads\(\);\s*(//.*)?$"), "tirx": re.compile(r"^\s*txl\.cuda\.cta_sync\(\)\s*(#.*)?$")}
_SHFL = re.compile(r"__shfl_xor_sync\(")
_SHFL_OFF = re.compile(r"(for\s*\(\s*int\s+\w+\s*=\s*)(\d+)(\s*;)")
_ASSIGN = {"cuda": re.compile(r"(?<![=!<>+\-*/])=(?!=)|\+=|-=|\*=|/="), "tirx": re.compile(r"\bassign\(|(?<![=!<>])=(?!=)")}
_MATH = {"cuda": re.compile(r"\bfmaxf\b|\bfminf\b"), "tirx": re.compile(r"txl\.max\b|txl\.min\b")}
_INIT = re.compile(r"-INFINITY|-FLT_MAX|-3\.4028\d*e\+?38f?|(?<![\w.])0\.0f?(?![\w.])")
_DECL = re.compile(r"\b(?:float|int)\s+(\w+)\s*(?:=|;)")


def _after_headers(line: str) -> int:
    """Offset of the statement after leading for/if/while headers on a CUDA line (0 if none)."""
    off = 0
    while True:
        m = re.match(r"\s*(for|if|while)\s*\(", line[off:])
        if not m:
            return off
        depth, j = 0, off + m.end() - 1
        while j < len(line):
            depth += {"(": 1, ")": -1}.get(line[j], 0)
            j += 1
            if depth == 0:
                break
        off = j


def mutants(src: str, lang: str = "cuda", per_op: int = PER_OP) -> list[Mutant]:
    lines = src.splitlines()
    code = _code_lines(src, lang)
    out: list[Mutant] = []

    def add(op, accept, cands):
        for i, new, desc, *bug in _spread(cands, per_op):
            out.append(Mutant(op, i + 1, desc, _sub_line(src, i, new), accept, bug[0] if bug else True))

    # cmp
    c = []
    for i in code:
        part = _code_part(lines[i], lang)
        if _COND[lang].search(part):
            m = _CMP.search(part, _COND[lang].search(part).end())
            if m:
                flip = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}[m.group(1)]
                c.append((i, lines[i][:m.start()] + flip + lines[i][m.end():], f"{m.group(1)} -> {flip}"))
    add("cmp", ("indexing",), c)
    # idx1: subscript on a read (right of an assignment, or an argument)
    c = []
    for i in code:
        part = _code_part(lines[i], lang)
        for m in _SUBSCRIPT[lang].finditer(part):
            inner = m.group(m.lastindex)
            if inner.strip() in ("0",) or "+ 1" in inner:
                continue
            new = lines[i][:m.start(m.lastindex)] + f"{inner} + 1" + lines[i][m.end(m.lastindex):]
            decl = lang == "cuda" and re.search(r"(?:__shared__|\b(?:float|int|double))\s+[\w\[\]\s]*$", part[:m.start()])
            c.append((i, new, f"[{inner}] -> [{inner} + 1]" + (" (array declared 1 larger)" if decl else ""), not decl))
            break
    add("idx1", ("indexing",), c)
    # const
    c = []
    for i in code:
        part = _code_part(lines[i], lang)
        if lang == "cuda" and "__launch_bounds__" in part:
            continue
        for m in _INT.finditer(part):
            v = int(m.group(1))
            if v < 2 or v > 100000:
                continue
            nv = v // 2 if v & (v - 1) == 0 else v + 1
            c.append((i, lines[i][:m.start()] + str(nv) + lines[i][m.end():], f"{v} -> {nv}"))
            break
    add("const", STRUCTURAL, c)
    # define (CUDA tile constants; tirx module-level BLOCK = ...)
    c = []
    for i, line in enumerate(lines):
        m = _DEFINE.match(line) if lang == "cuda" else re.match(r"^(\s*(?:BLOCK|TILE)\w*\s*=\s*)(\d+)\s*$", line)
        if m:
            v = int(m.group(2))
            c.append((i, m.group(1) + str(2 * v) + line[m.end():], f"{v} -> {2 * v}"))
    add("define", ("tiling",), c)
    # nosync
    c = [(i, "", "barrier deleted") for i in code if _SYNC[lang].match(lines[i])]
    add("nosync", ("sync",), c)
    # shfl
    c, c2 = [], []
    for i in code:
        if _SHFL.search(lines[i]):
            c.append((i, _SHFL.sub("__shfl_down_sync(", lines[i], count=1), "__shfl_xor_sync -> __shfl_down_sync"))
        m = _SHFL_OFF.search(lines[i])
        if m and i + 1 < len(lines) and "__shfl" in lines[i] + lines[i + 1]:
            v = int(m.group(2))
            c2.append((i, lines[i][:m.start(2)] + str(v // 2) + lines[i][m.end(2):], f"shuffle offset {v} -> {v // 2}"))
    add("shfl", ("sync",), c)
    add("shfl_off", ("sync", "indexing"), c2)        # the edit is a loop start that controls which lanes exchange
    # aop: only float arithmetic (index / pointer arithmetic is an indexing or memory change, not numerical)
    if lang == "cuda":
        floats = set(re.findall(r"\bfloat\s+(\w+)", src)) | set(re.findall(r"__shared__\s+float\s+(\w+)", src))
    else:
        floats = set(re.findall(r"^\s*(\w+)\s*=\s*txl\.local_scalar\(\s*\"float", src, re.M))
    c = []
    for i in code:
        full = _code_part(lines[i], lang)
        off = _after_headers(full) if lang == "cuda" else 0           # 'for (...) sum += ...': the statement
        part = full[off:]
        a = _ASSIGN[lang].search(part)
        if not a or _COND[lang].match(part.strip()) or "Idx" in part or "size_t" in part:
            continue
        if lang == "cuda":
            lhs = re.findall(r"(\w+)\s*(?:\[[^\]]*\])?\s*$", part[:a.start()])
            rhs_start = a.end()
        else:
            m0 = re.match(r"\s*txl\.assign\(\s*(\w+)\s*,", part)
            lhs = [m0.group(1)] if m0 else []
            rhs_start = m0.end() if m0 else len(part)
        if not lhs or lhs[0] not in floats:
            continue
        if a.group(0) == "+=":
            c.append((i, lines[i][:off + a.start()] + "=" + lines[i][off + a.end():], "+= -> ="))
            continue
        rhs = re.sub(r"\[[^\[\]]*\]", lambda k: "#" * len(k.group(0)), part[rhs_start:])   # not inside subscripts
        m = re.search(r"(?<=[\w)\]\s#])([+*])(?=[\s\w(])", rhs)
        if m:
            j = off + rhs_start + m.start(1)
            flip = {"+": "-", "*": "/"}[m.group(1)]
            c.append((i, lines[i][:j] + flip + lines[i][j + 1:], f"{m.group(1)} -> {flip}"))
    add("aop", ("numerical",), c)
    # math
    c = []
    for i in code:
        m = _MATH[lang].search(_code_part(lines[i], lang))
        if m:
            t = m.group(0)
            flip = t.replace("max", "min") if "max" in t else t.replace("min", "max")
            c.append((i, lines[i][:m.start()] + flip + lines[i][m.end():], f"{t} -> {flip}"))
    add("math", ("numerical",), c)
    # init
    c = []
    for i in code:
        part = _code_part(lines[i], lang)
        m = _INIT.search(part)
        if m and re.search(r"=\s*" + re.escape(m.group(0)), part):
            zero, one = ("0.0f", "1.0f") if lang == "cuda" else ("0.0", "1.0")
            new = zero if m.group(0) not in ("0.0f", "0.0") else one
            c.append((i, lines[i][:m.start()] + new + lines[i][m.end():], f"{m.group(0)} -> {new}"))
    add("init", ("numerical",), c)
    # --- operators added AFTER the rule revision (held out) ---
    # negate: an if-condition negated                                          indexing
    c = []
    for i in code:
        part = _code_part(lines[i], lang)
        m = re.search(r"\bif\s*\(", part) if lang == "cuda" else re.search(r"txl\.if_\(", part)
        if m:
            end = m.end()
            depth, j = 1, end
            while j < len(part) and depth:
                depth += {"(": 1, ")": -1}.get(part[j], 0)
                j += 1
            cond = part[end:j - 1]
            neg = f"!({cond})" if lang == "cuda" else f"txl.logical_not({cond})"
            c.append((i, lines[i][:end] + neg + lines[i][j - 1:], f"if ({cond}) negated"))
    add("negate", ("indexing",), c)
    # smem: a shared-memory array size halved                                  memory, tiling
    c = []
    for i in code:
        part = _code_part(lines[i], lang)
        m = re.search(r"__shared__\s+float\s+\w+\s*\[\s*(\d+)\s*\]", part) if lang == "cuda" else \
            re.search(r"\.alloc\(\[\s*(\d+)", part)
        if m and int(m.group(1)) >= 2:
            v = int(m.group(1))
            c.append((i, lines[i][:m.start(1)] + str(v // 2) + lines[i][m.end(1):], f"shared size {v} -> {v // 2}"))
    add("smem", ("memory", "tiling"), c)
    # dtype: a float local made double (CUDA) / float16 (tirx)                numerical
    c = []
    for i in code:
        part = _code_part(lines[i], lang)
        m = re.search(r"(?<![\w*])float(?=\s+\w+\s*=)", part) if lang == "cuda" else re.search(r'"float32"', part)
        if m and "__shared__" not in part:
            new_t = "double" if lang == "cuda" else '"float16"'
            c.append((i, lines[i][:m.start()] + new_t + lines[i][m.end():], f"{m.group(0)} -> {new_t}",
                      lang != "cuda"))                       # float -> double only adds precision
    add("dtype", ("numerical",), c)
    # rename (no semantic change): a local float/int variable, whole-word, outside comments
    names = []
    for i in code:
        for m in _DECL.finditer(_code_part(lines[i], lang)):
            if m.group(1) not in names and len(m.group(1)) <= 12:
                names.append(m.group(1))
    if lang == "tirx":
        names = [n for n in re.findall(r"^\s+(\w+)\s*=\s*txl\.local_scalar", src, re.M)]
    for n in _spread(names, per_op):
        new_name = f"{n}_renamed"
        out.append(Mutant("rename", 0, f"{n} -> {new_name}", re.sub(rf"\b{re.escape(n)}\b", new_name, src), ()))
    # format (no semantic change)
    if lang == "cuda":
        fmt = []
        for i, line in enumerate(lines):
            if i not in code:
                fmt.append(line)
                continue
            body = _code_part(line, lang)
            comment = line[len(body):]
            body = re.sub(r";[ \t]+(?=\S)", ";\n    ", body)
            body = re.sub(r"\)\s*\{", ")\n{", body)
            fmt.append(body + comment)
        out.append(Mutant("format", 0, "statements re-wrapped (code only, comments untouched)", "\n".join(fmt) + "\n", ()))
    return out


def classify_mutant(workload: str, base_src: str, m: Mutant, lang: str, llm=None, classifier: str = "auto") -> dict:
    from ..validation.changeaware import plan
    p = plan(workload, base_src, m.src, lang, llm=llm, classifier=classifier) if llm else \
        plan(workload, base_src, m.src, lang, classifier=classifier)
    cats = list(p.cats)
    if m.accept:
        verdict = "hit" if set(cats) & set(m.accept) else ("none" if not cats else "wrong")
    else:
        verdict = "clean" if not cats else "spurious"
    info = getattr(p, "llm", None) or {}
    an = p.analysis or {}
    return {"op": m.op, "line": m.line, "desc": m.desc, "accept": list(m.accept), "bug": m.bug, "categories": cats,
            "verdict": verdict, "llm": info, "n_tests": len(p.tests),
            "analysis": {"mode": an.get("mode"), "ir_blind": an.get("ir_blind", False),
                         "fallback_reason": (an.get("fallback_reason") or None) and str(an["fallback_reason"])[:200]}}


def _detect(workload: str, base_label: str, base_src: str, lang: str, m: Mutant, device: str, volta: bool,
            llm=None, classifier: str = "auto") -> dict:
    """Is the mutant caught by A (fixed tests) and by C (fixed + change-aware tests [+ VOLTA])?"""
    from ..validation.changeaware.validator import ChangeAwareValidator
    from ..validation.fixed_tests.validator import FixedTestsValidator
    from ..validation.interface import Candidate, ValidationContext
    from ..validation.pipeline import run_pipeline
    parent = Candidate(workload, lang, base_src, label=base_label)
    cand = Candidate(workload, lang, m.src, parent=parent, label=f"{base_label}_{m.op}_{m.line}")
    ctx = ValidationContext(device=device, extra={"classifier": classifier, **({"llm_classifier": llm} if llm else {})})
    t0 = time.perf_counter()
    fixed = FixedTestsValidator().validate(cand, ctx)
    out = {"A": {"caught": not fixed.passed, "seconds": fixed.seconds}}
    if volta:
        r = run_pipeline(cand, "C", ctx)
        out["C"] = {"caught": not r.passed, "seconds": r.seconds}
    elif not fixed.passed:                                 # C runs the fixed tests too: already caught
        out["C"] = {"caught": True, "seconds": fixed.seconds, "changeaware_only": False}
    else:
        ca = ChangeAwareValidator().validate(cand, ctx)
        out["C"] = {"caught": not ca.passed, "seconds": fixed.seconds + ca.seconds, "changeaware_only": not ca.passed,
                    "failed_tests": [f.evidence.get("tag", f.message)[:120] for f in ca.findings if f.severity == "error"][:5]}
    out["seconds"] = time.perf_counter() - t0
    return out


def _base_ok(workload: str, base_label: str, base_src: str, lang: str, device: str) -> tuple[bool, str]:
    """A mutant of a kernel that is already wrong is "caught" by anything, so detection only counts kernels that
    pass the fixed tests and the always-on change-aware floor tests themselves."""
    from ..validation.changeaware.validator import ChangeAwareValidator
    from ..validation.fixed_tests.validator import FixedTestsValidator
    from ..validation.interface import Candidate, ValidationContext
    ctx = ValidationContext(device=device)
    parent = Candidate(workload, lang, base_src, label=base_label)
    cand = Candidate(workload, lang, base_src, parent=parent, label=base_label)     # no change: floor tests only
    for v in (FixedTestsValidator(), ChangeAwareValidator()):
        r = v.validate(cand, ctx)
        if not r.passed:
            errs = [f.message for f in r.findings if f.severity == "error"]
            return False, f"{r.validator}: {errs[0][:200] if errs else 'failed'}"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="akc mutants", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-op", type=int, default=PER_OP)
    ap.add_argument("--only", help="comma-separated base labels")
    ap.add_argument("--set", default="bases", choices=["bases", "heldout", "all"],
                    help="bases: kernels the rules were revised on; heldout: kernels never looked at while revising")
    ap.add_argument("--llm-classify", action="store_true", help="also ask an LLM to classify (spends Lumen coins)")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--detect", action="store_true", help="run configurations A and C on every mutant (GPU)")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--volta", action="store_true", help="with --detect: include VOLTA in C (slow)")
    ap.add_argument("--classifier", default="ir", choices=["ir", "ast"],
                    help="tirx-lite rows: IR change analysis (default) or the Python-AST classifier")
    ap.add_argument("--compare-with", default=None, help="another mutants JSON: print an agreement matrix")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    llm = None
    if a.llm_classify:
        from ..validation.changeaware.llm_classify import LlmClassifier
        llm = LlmClassifier(**({"model": a.llm_model} if a.llm_model else {}))
    rows, t0 = [], time.perf_counter()
    out = Path(a.out) if a.out else RESULTS / "eval" / f"mutants_{'llm_' if llm else ''}{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save = lambda done: out.write_text(json.dumps({"rows": rows, "seconds": time.perf_counter() - t0,  # noqa: E731
                                                   "llm_model": getattr(llm, "model", None), "classifier": a.classifier,
                                                   "complete": done}, indent=1))
    sets = {"bases": BASES, "heldout": HELDOUT, "all": BASES + HELDOUT}[a.set]
    for label, workload, rel in sets:
        if a.only and label not in a.only.split(","):
            continue
        path = ROOT / rel
        if not path.exists():
            print(f"skip {label}: {rel} not found")
            continue
        src, lang = path.read_text(), ("tirx" if path.suffix == ".py" else "cuda")
        ok, why = _base_ok(workload, label, src, lang, a.device) if a.detect else (True, "")
        if not ok:
            print(f"  {label}: the base kernel itself fails ({why}); its mutants are classified but not run", flush=True)
        muts = mutants(src, lang, a.per_op)
        if lang == "tirx" and a.classifier == "ir":          # trace all mutants of this base in one child process
            from ..validation.changeaware import ir_worker
            ir_worker.prefetch(workload, [(src, m.src, ir_worker.trace_shapes(workload, src, m.src)) for m in muts
                                          if src != m.src], timeout_s=ir_worker.TIMEOUT_S)
        for m in muts:
            r = {"base": label, "workload": workload, "lang": lang,
                 **classify_mutant(workload, src, m, lang, llm, "ast" if a.classifier == "ast" else "auto")}
            if a.detect:
                r["base_ok"] = ok
                if ok:
                    r["detect"] = _detect(workload, label, src, lang, m, a.device, a.volta, llm,
                                          "ast" if a.classifier == "ast" else "auto")
            rows.append(r)
        save(False)                                   # partial results survive a Slurm time limit
        print(f"  {label}: done ({len(rows)} mutants so far, {time.perf_counter() - t0:.0f}s)", flush=True)
    _report(rows, a.llm_classify)
    save(True)
    print(f"saved {out}")
    if a.compare_with:
        other = json.loads(Path(a.compare_with).read_text())["rows"]
        print(f"\nagreement with {a.compare_with} (rows: that file's verdict, columns: this run's):")
        print(agreement_text(other, rows))
    return 0


VERDICTS = ("hit", "wrong", "none", "clean", "spurious")


def _mutant_key(r: dict) -> tuple:
    return (r["base"], r["op"], r["line"], r["desc"])


def agreement(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Verdict pairs for the mutants present in both runs: {"matrix": {(va, vb): n}, "n": shared, "only_a", "only_b",
    "summary_a"/"summary_b": summarize() of the shared rows, "ir_blind_b": rows of b kept on AST labels}."""
    a = {_mutant_key(r): r for r in rows_a}
    b = {_mutant_key(r): r for r in rows_b}
    shared = [k for k in a if k in b]
    matrix: dict[tuple, int] = {}
    for k in shared:
        key = (a[k]["verdict"], b[k]["verdict"])
        matrix[key] = matrix.get(key, 0) + 1
    return {"matrix": matrix, "n": len(shared), "only_a": len(a) - len(shared), "only_b": len(b) - len(shared),
            "summary_a": summarize([a[k] for k in shared])["total"], "summary_b": summarize([b[k] for k in shared])["total"],
            "ir_blind_b": sum(bool((b[k].get("analysis") or {}).get("ir_blind")) for k in shared),
            "fallback_b": sum((b[k].get("analysis") or {}).get("mode") == "ast-fallback" for k in shared)}


def agreement_text(rows_a: list[dict], rows_b: list[dict]) -> str:
    ag = agreement(rows_a, rows_b)
    used = [v for v in VERDICTS if any(v in k for k in ag["matrix"])]
    lines = [f"{'':10s}" + "".join(f"{v:>9s}" for v in used)]
    for va in used:
        lines.append(f"{va:10s}" + "".join(f"{ag['matrix'].get((va, vb), 0):9d}" for vb in used))
    sa, sb = ag["summary_a"], ag["summary_b"]
    lines.append(f"shared mutants {ag['n']} (only in the other file {ag['only_a']}, only here {ag['only_b']}); "
                 f"hit {sa['hit']} -> {sb['hit']} / {sa['semantic']}, wrong {sa['wrong']} -> {sb['wrong']}, "
                 f"none {sa['none']} -> {sb['none']}, spurious {sa['spurious']} -> {sb['spurious']} / {sa['noop']}; "
                 f"here: ir_blind {ag['ir_blind_b']}, AST fallback {ag['fallback_b']}")
    return "\n".join(lines)


def summarize(rows: list[dict]) -> dict:
    by_op: dict[str, dict] = {}
    for r in rows:
        d = by_op.setdefault(r["op"], {"n": 0, "hit": 0, "wrong": 0, "none": 0, "clean": 0, "spurious": 0})
        d["n"] += 1
        d[r["verdict"]] += 1
    sem = [r for r in rows if r["accept"]]
    noop = [r for r in rows if not r["accept"]]
    tot = {"semantic": len(sem), "hit": sum(r["verdict"] == "hit" for r in sem),
           "wrong": sum(r["verdict"] == "wrong" for r in sem), "none": sum(r["verdict"] == "none" for r in sem),
           "noop": len(noop), "spurious": sum(r["verdict"] == "spurious" for r in noop)}
    return {"by_op": by_op, "total": tot}


def _report(rows: list[dict], llm: bool) -> None:
    s = summarize(rows)
    print(f"{'operator':8s} {'n':>3s} {'hit':>4s} {'wrong':>5s} {'none':>4s}   (no-op ops: clean / spurious)")
    for op, d in s["by_op"].items():
        if d["clean"] + d["spurious"]:
            print(f"{op:8s} {d['n']:3d}   clean {d['clean']}  spurious {d['spurious']}")
        else:
            print(f"{op:8s} {d['n']:3d} {d['hit']:4d} {d['wrong']:5d} {d['none']:4d}")
    t = s["total"]
    print(f"semantic mutants: {t['hit']}/{t['semantic']} right category, {t['wrong']} wrong, {t['none']} no category; "
          f"no-op mutants: {t['spurious']}/{t['noop']} spurious")
    if any("detect" in r for r in rows):
        det = [r for r in rows if "detect" in r and r["accept"] and r.get("bug", True)]
        a = sum(r["detect"]["A"]["caught"] for r in det)
        c = sum(r["detect"]["C"]["caught"] for r in det)
        fa = [r for r in rows if "detect" in r and not r["accept"]]
        skipped = sorted({r["base"] for r in rows if r.get("base_ok") is False})
        print(f"detection (mutants that can change outputs; some may still be equivalent): A {a}/{len(det)}, "
              f"C {c}/{len(det)}; false alarms on no-op edits: A {sum(r['detect']['A']['caught'] for r in fa)}/{len(fa)}, "
              f"C {sum(r['detect']['C']['caught'] for r in fa)}/{len(fa)}"
              + (f"; bases that fail by themselves (not run): {', '.join(skipped)}" if skipped else ""))
        for r in det:
            if not r["detect"]["C"]["caught"]:
                print(f"  C missed: {r['base']} {r['op']} L{r['line']} {r['desc']} (categories {r['categories']})")
    if llm:
        cost = sum((r.get("llm") or {}).get("cost", 0) for r in rows)
        print(f"LLM cost: {cost:.4f} coins")
