"""The generated edits of the detection workload: the 34 hard edits, and the archive.

The final A100 run (2026-11-17, results/eval/mutants_detect_tirx_{bases,heldout,fresh}.json) ran all 164 generated
tirx-lite edits plus 27 renames. Most edits say nothing about configurations A / B / C:
  - 119 are caught by plain fixed tests in every configuration (36 do not even compile);
  - 11 cannot change any output (NO_CHANGE: block / tile / unroll constants a kernel derives everything from, and
    GEMM padding values that are always multiplied by zero).
Those 130 are archived: `akc mutants --detect` skips them unless --archived is given (it also archives the legacy
CUDA edits). The other 34 are the hard edits (HARD): they change the kernel's behaviour and fixed tests miss all of
them. The 27 renames stay in the workload as false-alarm controls. What each edit does: notes/detection_workload.md.

  HARD, NO_CHANGE                    frozensets of (base, op, line, desc) keys, as in a mutants row
  key(row_or_label, mutant=None)     the key of a mutants row, or of (base label, Mutant)
  in_workload(label, lang, m)        is this mutant run by default with --detect? (HARD edits + tirx-lite no-op edits)
  select_hard(rows) -> rows          from a --detect run that includes A0: edits not caught by A0 and not in NO_CHANGE
  summary(rows) -> {config: caught}  caught counts over the given rows

  python -m akc.evaluation.hard_edits   rebuild results/eval/hard_edits.json from the full run and check it gives HARD

The keys hold line numbers, so editing a base kernel can shift them; tests/evaluation/test_hard_edits.py checks
that the generator still produces every key.
"""
from __future__ import annotations

import json
import sys

from ..paths import RESULTS

FULL_RUN = [RESULTS / "eval" / f"mutants_detect_tirx_{s}.json" for s in ("bases", "heldout", "fresh")]
OUT = RESULTS / "eval" / "hard_edits.json"
CONFIGS = ("A0", "A", "B", "C")

HARD = frozenset([
    ("softmax_txl_ref", "cmp", 30, "< -> <="),
    ("softmax_txl_ref", "cmp", 43, "< -> <="),
    ("softmax_txl_ref", "cmp", 51, "< -> <="),
    ("softmax_txl_ref", "const", 8, "128 -> 64"),
    ("softmax_txl_ref", "nosync", 37, "barrier deleted"),
    ("softmax_txl_ref", "math", 46, "txl.max -> txl.min"),
    ("glm_txl_C_r2", "cmp", 95, "< -> <="),
    ("glm_txl_C_r2", "init", 4, "-3.4028234663852886e38 -> 0.0"),
    ("glm_txl_C_r2", "negate", 70, "if (lane == 0) negated"),
    ("glm_txl_D1_r2", "cmp", 148, "< -> <="),
    ("glm_txl_D1_r2", "negate", 63, "if (lane == 0) negated"),
    ("glm_txl_D2_r1", "cmp", 151, "< -> <="),
    ("glm_txl_D2_r1", "negate", 127, "if (lane == 0) negated"),
    ("glm_txl_D2_r1", "negate", 138, "if (lane == 0) negated"),
    ("glm_txl_D3_r1", "cmp", 97, "< -> <="),
    ("glm_txl_D3_r1", "const", 65, "16 -> 8"),
    ("glm_txl_D3_r1", "init", 12, "-3.4028234663852886e38 -> 0.0"),
    ("glm_txl_D3_r1", "negate", 109, "if (lane == 0) negated"),
    ("glm_txl_D3_r1", "negate", 141, "if (lane == 0) negated"),
    ("layernorm_txl_opt", "cmp", 36, "< -> <="),
    ("layernorm_txl_opt", "cmp", 50, "< -> <="),
    ("layernorm_txl_opt", "cmp", 60, "< -> <="),
    ("layernorm_txl_opt", "const", 11, "128 -> 64"),
    ("layernorm_txl_opt", "nosync", 43, "barrier deleted"),
    ("layernorm_txl_opt", "aop", 64, "+ -> -"),
    ("layernorm_txl_opt", "smem", 25, "shared size BLOCK -> BLOCK // 2"),
    ("gemm_txl_ref", "cmp", 27, "< -> <="),
    ("gemm_txl_ref", "cmp", 28, "< -> <="),
    ("layernorm_txl_ref", "idx1", 36, "[r, 0] -> [r, 0 + 1]"),
    ("layernorm_txl_ref", "negate", 35, "if (tid == 0) negated"),
    ("gemm_txl_opt", "cmp", 39, "< -> <="),
    ("gemm_txl_opt", "cmp", 44, "< -> <="),
    ("gemm_txl_opt", "cmp", 54, "< -> <="),
    ("gemm_txl_opt", "smem", 31, "shared size TILE -> TILE // 2"),
])

# Edits that cannot change any output. VOLTA proves the GLM ones and the two GEMM padding ones equivalent; it cannot
# decide the four GEMM tile ones (block 0 then covers another part of C), which were judged by reading the code.
NO_CHANGE = frozenset([
    ("glm_txl_C_r2", "const", 9, "128 -> 64"),
    ("glm_txl_C_r2", "const", 11, "4 -> 2"),
    ("glm_txl_C_r2", "define", 9, "128 -> 256"),
    ("glm_txl_D2_r1", "const", 19, "128 -> 64"),
    ("glm_txl_D2_r1", "define", 19, "128 -> 256"),
    ("gemm_txl_ref", "const", 10, "16 -> 8"),
    ("gemm_txl_ref", "define", 10, "16 -> 32"),
    ("gemm_txl_opt", "const", 11, "16 -> 8"),
    ("gemm_txl_opt", "define", 11, "16 -> 32"),
    ("gemm_txl_opt", "init", 38, "txl.float32(0.0) -> txl.float32(1.0)"),
    ("gemm_txl_opt", "init", 43, "txl.float32(0.0) -> txl.float32(1.0)"),
])


def key(row_or_label, mutant=None) -> tuple:
    if mutant is not None:
        return (row_or_label, mutant.op, mutant.line, mutant.desc)
    r = row_or_label
    return (r["base"], r["op"], r["line"], r["desc"])


def in_workload(label: str, lang: str, m) -> bool:
    """Run by default with --detect: a hard edit, or a tirx-lite edit that changes nothing (a false-alarm control)."""
    return lang == "tirx" and (key(label, m) in HARD or not m.accept)


def select_hard(rows: list[dict]) -> list[dict]:
    """Edits (not no-op controls) that plain fixed tests (A0) miss, minus the edits that cannot change any output."""
    return [r for r in rows if r["accept"] and r.get("bug", True) and "detect" in r
            and not r["detect"]["A0"]["caught"] and key(r) not in NO_CHANGE]


def summary(rows: list[dict]) -> dict:
    return {c: sum(bool(r["detect"].get(c, {}).get("caught")) for r in rows) for c in CONFIGS
            if any(c in r["detect"] for r in rows)}


def load_full_run() -> list[dict]:
    return [r for p in FULL_RUN for r in json.loads(p.read_text())["rows"]]


def main(argv: list[str] | None = None) -> int:
    rows = load_full_run()
    hard = select_hard(rows)
    keys = {key(r) for r in hard}
    if keys != HARD:
        print(f"selection differs from HARD: extra {sorted(keys - HARD)}, missing {sorted(HARD - keys)}")
        return 1
    edits = [r for r in rows if r["accept"]]
    noop = [r for r in rows if not r["accept"]]
    out = {"source": [str(p.relative_to(RESULTS.parent)) for p in FULL_RUN],
           "rule": "generated edits not caught by A0 (plain fixed tests), minus NO_CHANGE",
           "counts": {"edits_in_full_run": len(edits), "caught_by_A0": sum(r["detect"]["A0"]["caught"] for r in edits),
                      "no_change": len(NO_CHANGE), "hard": len(hard), "no_op_controls": len(noop)},
           "caught": summary(hard), "false_alarms": summary(noop),
           "rows": [{k: r[k] for k in ("base", "set", "workload", "op", "line", "desc", "categories")}
                    | {"caught": {c: bool(r["detect"][c]["caught"]) for c in CONFIGS},
                       "kinds": {c: {v: pv.get("kinds", []) for v, pv in r["detect"][c]["per_validator"].items()
                                     if pv.get("caught")} for c in CONFIGS}}
                    for r in hard]}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"hard edits: {len(hard)}; caught " + ", ".join(f"{c} {n}" for c, n in out["caught"].items())
          + f"; false alarms on {len(noop)} no-op edits: " + ", ".join(f"{c} {n}" for c, n in out["false_alarms"].items()))
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
