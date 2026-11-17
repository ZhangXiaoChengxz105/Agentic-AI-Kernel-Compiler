"""VOLTA on tirx-lite kernels (Tasks 1 and 3): which tirx kernels can VOLTA decide, and does it say the right thing?

Kernels of one workload (--workload, default softmax), all compared with the workload's tirx reference
(manifest references.tirx) by VoltaValidator's tirx path:
  reference  the tirx reference vs itself                                 expect EQUIV
  opt        the bug-free tirx candidates (layernorm_txl_opt, gemm_txl_opt) expect their manifest expect_volta
  bug        the injected tirx bugs (manifest variants with lang "tirx")  expect their own manifest expect_volta;
                                                                          without one, the CUDA counterpart's
                                                                          (<op>_bug_<same suffix>)
softmax only (its GLM pilot outputs and mutation-benchmark rows exist for softmax only):
  glm        the 4 GLM-written tirx kernels that pass the tests           expect EQUIV
  mutant     tirx mutants that configurations A and C's tests missed       (no expectation; `focus` marks the
             (results/eval/mutants_detect_{bases,heldout}.json, rows       races named in the README: `if (lane
             with lang tirx, bug true, A and C not caught; renames are     == 0)` negated, and `tid < s` ->
             no-op edits and are left out), regenerated with               `tid <= s` in softmax_txl_ref)
             mutants(src, "tirx", 2)
Shapes: the manifest's VOLTA default and deep shapes plus the change-aware VOLTA shapes for tiling and sync
changes (changeaware.generate.volta_shapes: sizes where CTA 0 crosses a tile boundary; capped by the manifest's
max_N), or --shapes ("256,200" = N values, or "M=12,N=12,K=40;M=32,N=32,K=48"). The Task 3 softmax run
(results/eval/volta_tirx.json) used --shapes 256,584,200. Coverage = decided runs / all runs (decided: EQUIV or
an error verdict; not UNSUPPORTED, TIMEOUT, or a harness problem), also per kernel (decided at every shape).

Ablation (--ablation): the reference and the GLM kernels at the first shape, once per rewrite with that rewrite
switched off, to show which construct each rewrite makes readable (VOLTA's own error message is kept).

Usage (CPU only; the full set takes a few minutes, so run it in a Slurm job, not on the login node):
  python scripts/experiments/volta_tirx.py [--workload softmax] [--shapes 256,200] [--groups reference,bug,glm,mutant]
                                          [--out PATH]
  python scripts/experiments/volta_tirx.py --workload gemm --groups reference,opt,bug
  python scripts/experiments/volta_tirx.py --ablation [--shapes 256]          (softmax: reference + GLM kernels)
Output: results/eval/volta_tirx.json (softmax) or volta_tirx_<workload>.json (rows: kernel, group, per-shape
        status / seconds / rewrites; summary); results/eval/volta_tirx_ablation.json (--ablation)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.mutants import BASES, HELDOUT, mutants  # noqa: E402
from akc.paths import RESULTS, ROOT  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402
from akc.validation.interface import Candidate, ValidationContext  # noqa: E402
from akc.validation.volta.layouts import expected_statuses  # noqa: E402
from akc.validation.volta.tirx import REWRITES  # noqa: E402
from akc.validation.volta.validator import VoltaValidator  # noqa: E402

WL = "softmax"                 # set by --workload
GLM = ("glm_txl_C_r2", "glm_txl_D1_r2", "glm_txl_D2_r1", "glm_txl_D3_r1")
DETECT = ("results/eval/mutants_detect_bases.json", "results/eval/mutants_detect_heldout.json")
DECIDED = ("EQUIV", "NOT_EQUIV", "RACE", "DEADLOCK", "OUT_OF_BOUNDS", "UNINIT", "FOOTPRINT", "DIV_BY_ZERO")


def _focus(base: str, op: str, line: int, desc: str) -> bool:
    return (op == "negate" and "lane == 0" in desc) or (base == "softmax_txl_ref" and op == "cmp" and line == 30)


def kernels(groups: set[str]) -> list[dict]:
    wl = load_workload(WL)
    ref = wl.reference("tirx")
    out = []
    if "reference" in groups:
        out.append({"kernel": ref.name, "group": "reference", "src": ref.file.read_text(), "expect": "EQUIV"})
    if "opt" in groups:
        for v in wl.candidates("tirx"):
            if not v.bug:
                out.append({"kernel": v.name, "group": "opt", "src": v.file.read_text(), "expect": v.expect_volta})
    if "bug" in groups:
        for v in wl.bugs("tirx"):
            twin = v.name.replace("_txl", "")
            cuda = next((c for c in wl.variants if c.name == twin and c.lang == "cuda"), None)
            out.append({"kernel": v.name, "group": "bug", "src": v.file.read_text(), "bug": v.bug,
                        "expect": v.expect_volta or (cuda.expect_volta if cuda else None),
                        "cuda_twin": twin if cuda else None, "cuda_expect": cuda.expect_volta if cuda else None})
    if WL != "softmax":
        return out                     # GLM kernels and mutation-benchmark rows exist for softmax only
    where = {label: rel for label, _, rel in BASES + HELDOUT}
    if "glm" in groups:
        for label in GLM:
            out.append({"kernel": label, "group": "glm", "src": (ROOT / where[label]).read_text(), "expect": "EQUIV",
                        "path": where[label]})
    if "mutant" in groups:
        seen = set()
        for p in DETECT:
            for r in json.loads((ROOT / p).read_text())["rows"]:
                det = r.get("detect") or {}
                if (r["lang"] != "tirx" or not r.get("bug", True) or r["op"] == "rename" or not det
                        or det["A"]["caught"] or det["C"]["caught"]):
                    continue
                key = (r["base"], r["op"], r["line"], r["desc"])
                if key in seen:
                    continue
                seen.add(key)
                src = (ROOT / where[r["base"]]).read_text()
                m = next(m for m in mutants(src, "tirx", 2) if (m.op, m.line, m.desc) == (r["op"], r["line"], r["desc"]))
                out.append({"kernel": f"{r['base']}_{m.op}_L{m.line}", "group": "mutant", "src": m.src,
                            "base": r["base"], "op": m.op, "line": m.line, "desc": m.desc,
                            "focus": _focus(r["base"], m.op, m.line, m.desc)})
    return out


def ablation(shape: dict, timeout: float, out: Path) -> int:
    """Each rewrite switched off in turn, on the reference + GLM kernels at one shape."""
    ks = kernels({"reference", "glm"})
    rows, t0 = [], time.perf_counter()
    for drop in (None,) + REWRITES:
        validator = VoltaValidator(timeout_s=timeout, tirx_rewrites=tuple(r for r in REWRITES if r != drop))
        for k in ks:
            rep = validator.validate(Candidate(WL, "tirx", k["src"], label=f"{k['kernel']}_no_{drop}"),
                                     ValidationContext(volta_shapes=[shape], device="cpu"))
            run = (rep.artifacts.get("runs") or [{}])[0]
            rows.append({"dropped": drop, "kernel": k["kernel"], "status": run.get("status"),
                         "detail": run.get("detail", "")[:200], "decided": rep.decided})
            print(f"without {str(drop):11s} {k['kernel']:16s} {run.get('status')}: {run.get('detail', '')[:110]}",
                  flush=True)
    summary = {str(d): f"{sum(r['decided'] for r in rows if r['dropped'] == d)}/{len(ks)} decided"
               for d in (None,) + REWRITES}
    out.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "shape": shape,
                               "summary": summary, "rows": rows, "seconds": round(time.perf_counter() - t0, 1)},
                              indent=1))
    print(json.dumps(summary, indent=1))
    print(f"saved {out}")
    return 0


def parse_shapes(text: str) -> list[dict]:
    """"256,200" -> [{"N": 256}, {"N": 200}];  "M=12,N=12,K=40;M=32,N=32,K=48" -> two full shapes."""
    if "=" not in text:
        return [{"N": int(n)} for n in text.split(",")]
    return [{k.strip(): int(v) for k, v in (kv.split("=") for kv in chunk.split(","))}
            for chunk in text.split(";") if chunk.strip()]


def default_shapes(wl) -> list[dict]:
    from akc.validation.changeaware.generate import volta_shapes
    shapes = [wl.volta["default"]] + ([wl.volta["deep"]] if wl.volta.get("deep") else [])
    return shapes + volta_shapes(wl.op, ["tiling", "sync"])


def main(argv: list[str] | None = None) -> int:
    global WL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", default="softmax")
    ap.add_argument("--shapes", default=None, help='"256,200" (N values) or "M=12,N=12,K=40;..." (default: manifest '
                                                   "VOLTA default + deep + change-aware tiling/sync shapes)")
    ap.add_argument("--groups", default="reference,opt,bug,glm,mutant")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds per VOLTA run")
    ap.add_argument("--out", default=None, help="default results/eval/volta_tirx.json (_ablation.json)")
    ap.add_argument("--ablation", action="store_true", help="switch each rewrite off in turn (reference + GLM)")
    a = ap.parse_args(argv)
    WL = a.workload
    wl = load_workload(WL)
    shapes = wl.cap_volta_shapes(parse_shapes(a.shapes) if a.shapes else default_shapes(wl))
    if a.ablation:
        return ablation(shapes[0], a.timeout, Path(a.out or RESULTS / "eval" / "volta_tirx_ablation.json"))
    validator = VoltaValidator(timeout_s=a.timeout)
    rows, t0 = [], time.perf_counter()
    for k in kernels(set(a.groups.split(","))):
        src = k.pop("src")
        cand = Candidate(WL, "tirx", src, label=k["kernel"])
        rep = validator.validate(cand, ValidationContext(volta_shapes=shapes, device="cpu"))
        runs = rep.artifacts.get("runs") or [{"shape": s, "status": f.kind} for s, f in zip(shapes, rep.findings)]
        per = []
        for run in runs:
            exp = expected_statuses(k.get("expect"), run["shape"])
            per.append({"N": run["shape"].get("N"), "shape": run["shape"], "status": run.get("status"),
                        "volta_s": round(run["wall_s"], 3) if run.get("wall_s") is not None else None,
                        "build_s": run.get("build_s"), "alone": (run.get("alone") or {}).get("status"),
                        "expect": list(exp) if exp else None,
                        "as_expected": (run.get("status") in exp) if exp else None,
                        "rewrites": (run.get("rewrites") or {}).get("candidate"),
                        "detail": run.get("detail", "")[:200]})
        hints = sorted({f.evidence.get("hint") for f in rep.findings if f.evidence.get("hint")})
        row = {**k, "runs": per, "decided": rep.decided, "caught": not rep.passed, "seconds": round(rep.seconds, 2),
               "hints": hints}
        rows.append(row)
        print(f"{k['group']:9s} {k['kernel']:34s} " + "  ".join(
            f"{','.join(f'{d}={x}' for d, x in p['shape'].items())}:{p['status']}"
            + ("" if p["as_expected"] in (None, True) else "(UNEXPECTED)") for p in per)
            + f"  ({rep.seconds:.1f}s)", flush=True)

    all_runs = [p for r in rows for p in r["runs"]]
    decided_runs = [p for p in all_runs if p["status"] in DECIDED]
    checked = [p for p in all_runs if p["as_expected"] is not None]
    summary = {
        "workload": WL,
        "shapes": shapes,
        "kernels": len(rows),
        "runs": len(all_runs),
        "decided_runs": len(decided_runs),
        "coverage_runs": round(len(decided_runs) / len(all_runs), 3) if all_runs else None,
        "decided_kernels": sum(r["decided"] for r in rows),
        "coverage_kernels": round(sum(r["decided"] for r in rows) / len(rows), 3) if rows else None,
        "as_expected": f"{sum(p['as_expected'] for p in checked)}/{len(checked)}",
        "statuses": {s: sum(p["status"] == s for p in all_runs) for s in sorted({p["status"] for p in all_runs})},
        "focus_mutants_caught": f"{sum(r['caught'] for r in rows if r.get('focus'))}/"
                                f"{sum(1 for r in rows if r.get('focus'))}",
        "volta_s_total": round(sum(p["volta_s"] or 0 for p in all_runs), 2),
        "build_s_total": round(sum(p["build_s"] or 0 for p in all_runs), 2),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out = Path(a.out or RESULTS / "eval" / ("volta_tirx.json" if WL == "softmax" else f"volta_tirx_{WL}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "summary": summary,
                               "rewrites": list(validator.tirx_rewrites), "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
