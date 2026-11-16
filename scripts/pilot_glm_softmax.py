#!/usr/bin/env python3
"""GLM pilot: can GLM-5.3 (via Lumen) write a correct row-wise softmax in CUDA and in tirx-lite?

A mini version of the Task 4 loop, per language:
    prompt -> GLM -> extract code -> evaluate on CPU -> if wrong, feed the evaluation back (<= --rounds total)
Evaluation (no GPU):
    CUDA : nvcc (sm_80) -> VOLTA vs kernels/softmax/softmax_ref at N = 256, 200, 100 (equivalence + races)
    TIRx : safety scan -> trace -> sm_80 compile -> NumSim on CPU vs float64 softmax (normal and x100 inputs)
           -> Racecheck                                   (agent/tirx_eval.py)
Budget: Lumen client guard (<= 0.5 coin/call, keep >= 1 coin) + --max-coins for the whole pilot.
Everything is saved under results/pilot/<timestamp>/ ; every Lumen call is in logs/lumen/calls.jsonl.

Usage: source env.sh && envs/tirx/bin/python scripts/pilot_glm_softmax.py [--langs cuda,tirx] [--rounds 3] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.lumen import LumenClient  # noqa: E402
from agent.tirx_eval import evaluate_isolated  # noqa: E402
from verify import WORKSPACE  # noqa: E402
from verify.cases import Variant, load_manifest, make_case  # noqa: E402
from verify.ptx import CompileError, compile_to_ptx  # noqa: E402
from verify.hints import hint_for  # noqa: E402
from verify.volta import run_compare  # noqa: E402

TXL_DIR = WORKSPACE / "third_party" / "tirx-kernels" / "tirx_kernels"
VOLTA_NS = (256, 200, 100)
TIRX_CASES = [{"rows": 2, "n": 200, "dist": "normal"}, {"rows": 2, "n": 200, "dist": "large"},
              {"rows": 2, "n": 256, "dist": "normal"}]

CUDA_SYSTEM = "You are an expert CUDA performance engineer. You write correct, fast kernels for NVIDIA GPUs."
CUDA_TASK = """Write ONE CUDA C++ kernel for an NVIDIA A100 (sm_80).

Task: row-wise softmax over a row-major float32 matrix x[rows, N]; output y[rows, N]:
    y[r, i] = exp(x[r, i] - max_j x[r, j]) / sum_j exp(x[r, j] - max_j x[r, j])

Requirements
- Exact signature: extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
- Launch: grid = rows (one CTA per row), blockDim.x = 128 exactly.
- Correct for ANY N >= 1 (N is often not a multiple of 128) and numerically stable for large |x| (e.g. x ~ 100).
- Write every exponential as EXP(v), a macro from "common.cuh" (you must #include "common.cuh");
  do not call expf/__expf directly. Our formal verifier needs this.
- Control flow may depend only on threadIdx/blockIdx/blockDim/N, never on data values
  (use fmaxf(a, b), not `if (a > b)`), because the verifier requires data-independent control flow.
- No dynamic shared memory, no cp.async, no cooperative groups, no external libraries.
- Aim for high performance (fast reductions, coalesced access).

Answer with exactly one ```cuda code block containing the complete .cu file and nothing else."""

TIRX_SYSTEM = ("You are an expert GPU kernel engineer. You write kernels in tirx-lite (txl), a traced Python DSL "
               "over the TIRx compiler that emits CUDA/PTX. You follow the documentation exactly and never invent APIs.")


def tirx_task() -> str:
    readme = (TXL_DIR / "tirx_lite" / "README.md").read_text()
    example = (TXL_DIR / "norm" / "rmsnorm.py").read_text()
    return f"""Write a tirx-lite kernel for an NVIDIA A100 (sm_80).

Task: row-wise softmax over a row-major float32 matrix x[rows, n]; output y[rows, n]:
    y[r, i] = exp(x[r, i] - max_j x[r, j]) / sum_j exp(x[r, j] - max_j x[r, j])

Required module interface (static shapes; one kernel is traced per (rows, n)):

```python
import tirx_kernels.tirx_lite as txl

def make_kernel(rows: int, n: int):
    @txl.kernel()
    def softmax_txl(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={{"cuda": {{"arch": "sm_80"}}}})
        ...  # your code: one CTA (128 threads) per row
    return softmax_txl
```

Requirements
- Correct for ANY n >= 1 (n is often not a multiple of 128) and numerically stable for large |x| (x ~ 100).
- Target is sm_80 (Ampere): do NOT use Blackwell/Hopper-only features (tcgen05, TMA/TensorMap, wgmma, clusters,
  packed f32x2 instructions such as mul.rz.ftz.f32x2, mbarrier pipelines). Plain f32 scalar ops, shared memory,
  CTA barriers and warp shuffles are fine.
- Only import tirx_kernels.tirx_lite (and math if needed). No benchmarking/testing code, no torch, no I/O.
- Aim for high performance.

Answer with exactly one ```python code block containing the complete module and nothing else.

=== tirx-lite authoring guide (README) ===
{readme}

=== Small verified sm_80 example (static shapes, global loads/stores, loops, guards) ===
```python
import tirx_kernels.tirx_lite as txl

def make_kernel(rows: int, n: int):
    @txl.kernel()
    def scale2(x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=rows, block=128),
                         backend_config={{"cuda": {{"arch": "sm_80"}}}})
        r = txl.cta_id()
        with txl.serial((n + 127) // 128) as it:
            col = it * 128 + txl.thread_id()
            with txl.if_(col < n), txl.then_():
                v = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))
                txl.ptx.st.global_.f32(y.ptr_to([r, col]), v * txl.float32(2.0))
    return scale2
```

=== Larger official example (RMSNorm; written for Blackwell, so ignore its f32x2 / sm_100-only parts and its
benchmark/test helpers, but its shared-memory, barrier and warp-shuffle idioms are valid on sm_80) ===
```python
{example}
```"""


def extract_code(text: str, lang: str) -> str | None:
    blocks = re.findall(r"```([A-Za-z+]*)\s*\n(.*?)```", text, re.S)
    want = ("cuda", "cpp", "c++", "c") if lang == "cuda" else ("python", "py")
    good = [b for tag, b in blocks if tag.lower() in want] or [b for _, b in blocks]
    return max(good, key=len).strip() + "\n" if good else None


# ---- evaluation ---------------------------------------------------------------------------------
def eval_cuda(path: Path) -> dict:
    _, variants = load_manifest("softmax")
    ref = next(v for v in variants if v.role == "reference")
    cand = Variant("softmax_glm", path, "candidate", (128,), None, None)
    out_dir = path.parent / "ptx"
    try:
        ref_ptx = compile_to_ptx(ref.file, out_dir).ptx
        b = compile_to_ptx(path, out_dir)
        compile_to_ptx(path, out_dir, mode="gpu")
    except CompileError as e:
        return {"stage": "compile", "error": e.stderr[-3000:], "volta": []}
    if "softmax_glm" not in b.kernels:
        return {"stage": "compile", "error": f"kernel softmax_glm not found; entries: {b.kernels}", "volta": []}
    runs = []
    for n in VOLTA_NS:
        r = run_compare(make_case("softmax", ref_ptx, ref, b.ptx, cand, {"N": n}, timeout_s=300))
        runs.append({"N": n, "status": r.status, "detail": r.detail[:300], "hint": hint_for(r), "mismatches": r.mismatches,
                     "examples": r.mismatch_examples[:3], "wall_s": round(r.wall_s or 0, 2)})
    bad = [x for x in runs if x["status"] != "EQUIV"]
    return {"stage": "ok" if not bad else "volta", "volta": runs}


def feedback_cuda(ev: dict) -> str:
    if ev["stage"] == "compile":
        return f"Your kernel did not compile (nvcc -arch=sm_80):\n{ev['error']}\nFix it and return the complete file."
    lines = []
    for x in ev["volta"]:
        if x["status"] == "EQUIV":
            continue
        if x["status"] == "NOT_EQUIV":
            lines.append(f"- N={x['N']}: VOLTA proved your output differs from the reference softmax "
                         f"({x['mismatches']} of {x['N']} elements, e.g. {', '.join(x['examples'])}).")
        elif x["status"] == "RACE":
            lines.append(f"- N={x['N']}: VOLTA found a data race: {x['detail']}")
        elif x["status"] == "UNSUPPORTED":
            lines.append(f"- N={x['N']}: the verifier cannot analyze the kernel: {x['detail']} "
                         "(control flow must not depend on data values).")
        else:
            lines.append(f"- N={x['N']}: verifier result {x['status']}: {x['detail']}")
        if x.get("hint"):
            lines.append(f"  hint: {x['hint']}")
    return ("Formal verification (VOLTA, one CTA, exact real arithmetic) of your kernel against a reference softmax "
            "failed:\n" + "\n".join(lines) + "\nFix the kernel and return the complete file.")


def eval_tirx(path: Path) -> dict:
    return evaluate_isolated(path, TIRX_CASES)


def feedback_tirx(ev: dict) -> str:
    if ev["stage"] in ("safety", "import", "timeout", "crash"):
        return f"Your module failed at stage '{ev['stage']}':\n{ev['error']}\nFix it and return the complete module."
    lines = []
    for c in ev["cases"]:
        tag = f"rows={c['rows']}, n={c['n']}, inputs={'N(0,1)' if c['dist'] == 'normal' else 'N(0,1)*100'}"
        if c.get("error"):
            lines.append(f"- {tag}: failed at stage '{c['stage']}':\n{c['error']}")
        elif c["stage"] == "wrong":
            if c.get("nonfinite"):
                lines.append(f"- {tag}: {c['nonfinite']} outputs are NaN/Inf (not numerically stable?)")
            if c.get("n_bad"):
                ex = c.get("example", {})
                lines.append(f"- {tag}: {c['n_bad']} outputs wrong vs float64 softmax, e.g. index {ex.get('index')}: "
                             f"got {ex.get('got')}, want {ex.get('want')} (max abs err {c.get('max_abs_err'):.3g})")
            if c.get("racecheck") not in (None, "clean"):
                lines.append(f"- {tag}: Racecheck reported a data race: {(c.get('race_findings') or [''])[0][:300]}")
    return ("CPU simulation (TIRx NumSim + Racecheck) of your kernel failed:\n" + "\n".join(lines) +
            "\nFix the kernel and return the complete module.")


LANGS = {
    "cuda": {"system": CUDA_SYSTEM, "task": lambda: CUDA_TASK, "ext": ".cu", "eval": eval_cuda, "feedback": feedback_cuda},
    "tirx": {"system": TIRX_SYSTEM, "task": tirx_task, "ext": ".py", "eval": eval_tirx, "feedback": feedback_tirx},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="cuda,tirx")
    ap.add_argument("--rounds", type=int, default=3, help="GLM calls per language (1 = no repair)")
    ap.add_argument("--max-tokens", type=int, default=100000,
                    help="output cap incl. reasoning; 24k was too small at full effort (reasoning alone used it all)")
    ap.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "none"],
                    help="forwarded to GLM via Lumen (default: maximum); 'none' = model default")
    ap.add_argument("--per-call-limit", type=float, default=1.0, help="Lumen client guard per call (coins)")
    ap.add_argument("--http-timeout", type=float, default=3600.0, help="seconds to wait for one reply")
    ap.add_argument("--max-coins", type=float, default=1.0, help="hard cap for the whole pilot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = LumenClient(model="glm-5.3", per_call_limit=args.per_call_limit, timeout_s=args.http_timeout)
    p_in, p_out = client.price()
    out = WORKSPACE / "results" / "pilot" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    spent, summary = 0.0, {"model": "glm-5.3", "rounds": args.rounds, "langs": {}}

    for lang in args.langs.split(","):
        L = LANGS[lang]
        msgs = [{"role": "system", "content": L["system"]}, {"role": "user", "content": L["task"]()}]
        hist = []
        for rnd in range(1, args.rounds + 1):
            est_in = LumenClient.estimate_tokens(msgs)
            worst = est_in / 1e6 * p_in + args.max_tokens / 1e6 * p_out
            print(f"[{lang} r{rnd}] prompt ~{est_in} tokens, worst-case {worst:.4f} coins, spent so far {spent:.4f}")
            if spent + worst > args.max_coins:
                print(f"[{lang}] stopping: pilot cap {args.max_coins} coins would be exceeded")
                break
            extra = {} if args.reasoning_effort == "none" else {"reasoning_effort": args.reasoning_effort}
            res = client.chat(msgs, max_tokens=args.max_tokens, dry_run=args.dry_run, tag=f"pilot-{lang}-r{rnd}", **extra)
            spent += res.cost
            (out / f"{lang}_r{rnd}_reply.md").write_text(
                f"# {lang} round {rnd}\n\nmodel={res.model} in={res.input_tokens} out={res.output_tokens} "
                f"cost={res.cost:.5f} latency={res.latency_s:.1f}s finish={res.finish_reason} "
                f"effort={args.reasoning_effort} usage={json.dumps((res.raw or {}).get('usage'))}\n\n"
                f"## reasoning\n{res.reasoning}\n\n## content\n{res.content}\n")
            print(f"[{lang} r{rnd}] reply: in={res.input_tokens} out={res.output_tokens} cost={res.cost:.4f} "
                  f"latency={res.latency_s:.0f}s finish={res.finish_reason}")
            if args.dry_run:
                break
            code = extract_code(res.content, lang)
            if code is None:
                ev = {"stage": "no_code", "error": f"no fenced code block in the reply (finish={res.finish_reason})"}
                fb = ("Your previous answer hit the output length limit before any code was written. "
                      "Output the complete file in one code block now."
                      if res.finish_reason == "length" else
                      "Your answer did not contain a fenced code block. Return the complete file in one code block.")
            else:
                path = out / f"{'softmax_glm' if lang == 'cuda' else 'softmax_txl'}_r{rnd}{L['ext']}"
                path.write_text(code)
                ev = L["eval"](path)
                fb = L["feedback"](ev) if ev["stage"] != "ok" else ""
            (out / f"{lang}_r{rnd}_eval.json").write_text(json.dumps(ev, indent=1, default=str))
            hist.append({"round": rnd, "stage": ev["stage"], "cost": res.cost, "in": res.input_tokens,
                         "out": res.output_tokens, "latency_s": round(res.latency_s, 1)})
            print(f"[{lang} r{rnd}] evaluation: {ev['stage']}")
            if ev["stage"] == "ok":
                break
            msgs += [{"role": "assistant", "content": res.content}, {"role": "user", "content": fb}]
        summary["langs"][lang] = {"history": hist, "passed": bool(hist and hist[-1]["stage"] == "ok")}
    summary["spent_coins"] = spent
    summary["reasoning_effort"] = args.reasoning_effort
    summary["max_tokens"] = args.max_tokens
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nspent {spent:.4f} coins; results in {out.relative_to(WORKSPACE)}")
    print(json.dumps(summary["langs"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
