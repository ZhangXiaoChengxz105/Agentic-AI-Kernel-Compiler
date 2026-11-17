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

Usage: akc pilot [--langs cuda,tirx] [--rounds 3] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from ..feedback.render import render
from ..llm.lumen import LumenClient
from ..paths import RESULTS, THIRD_PARTY
from ..task.testcase import TestCase
from ..task.workload import load_workload
from ..validation.fixed_tests.validator import FixedTestsValidator
from ..validation.interface import Candidate, ValidationContext
from ..validation.volta.validator import VoltaValidator

TXL_DIR = THIRD_PARTY / "tirx-kernels" / "tirx_kernels"
VOLTA_SHAPES = [{"N": 256}, {"N": 200}, {"N": 100}]
TIRX_TESTS = [TestCase("softmax", {"rows": 2, "N": 200}, "normal", 0, 1, "pilot"),
              TestCase("softmax", {"rows": 2, "N": 200}, "large", 0, 1, "pilot"),
              TestCase("softmax", {"rows": 2, "N": 256}, "normal", 0, 1, "pilot")]

CUDA_SYSTEM = "You are an expert CUDA performance engineer. You write correct, fast kernels for NVIDIA GPUs."
CUDA_TASK = """Write ONE CUDA C++ kernel for an NVIDIA A100 (sm_80).

Task: row-wise softmax over a row-major float32 matrix x[rows, N]; output y[rows, N]:
    y[r, i] = exp(x[r, i] - max_j x[r, j]) / sum_j exp(x[r, j] - max_j x[r, j])

{contract}

Requirements
- Exact signature: extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
- Launch: grid = rows (one CTA per row), blockDim.x = 128 exactly.
- Correct for every shape in the range above (N is often not a multiple of 128) and numerically stable for
  large |x| (e.g. x ~ 100).
- Write every exponential as EXP(v), a macro from "common.cuh" (you must #include "common.cuh");
  do not call expf/__expf directly. Our formal verifier needs this.
- Control flow may depend only on threadIdx/blockIdx/blockDim/N, never on data values
  (use fmaxf(a, b), not `if (a > b)`), because the verifier requires data-independent control flow.
- No dynamic shared memory, no cp.async, no cooperative groups, no external libraries.
- Aim for high performance (fast reductions, coalesced access).

Answer with exactly one ```cuda code block containing the complete .cu file and nothing else."""

TIRX_SYSTEM = ("You are an expert GPU kernel engineer. You write kernels in tirx-lite (txl), a traced Python DSL "
               "over the TIRx compiler that emits CUDA/PTX. You follow the documentation exactly and never invent APIs.")


def contract_text(wl, lang: str) -> str:
    """The CAKE shape contract for the agent's prompt: the declared shape range (correctness is checked across
    shapes and input distributions in it) and the one benchmark shape (speed)."""
    if wl.domain:
        lines = [f"Shape contract: the kernel must be correct for EVERY shape with {wl.domain_text()} (the declared "
                 "shape domain). Correctness is checked across different shapes and input distributions in this "
                 "range, including sizes that are not multiples of your tile or block sizes."]
    else:
        lines = ["Shape contract: no shape range is declared; the kernel must be correct for every shape."]
    if lang == "tirx":
        lines.append("make_kernel is traced once per shape (static shapes), so it must build a correct kernel for "
                     "every shape in the range, not only the benchmark shape.")
    if wl.perf_shape:
        bench = ", ".join(f"{k}={v}" for k, v in wl.perf_shape.items())
        lines.append(f"Speed is measured at {bench}: tune for that shape, but stay correct on the whole range.")
    return "\n".join(lines)


def cuda_task() -> str:
    return CUDA_TASK.format(contract=contract_text(load_workload("softmax"), "cuda"))


def tirx_task() -> str:
    readme = (TXL_DIR / "tirx_lite" / "README.md").read_text()
    example = (TXL_DIR / "norm" / "rmsnorm.py").read_text()
    contract = contract_text(load_workload("softmax"), "tirx")
    return f"""Write a tirx-lite kernel for an NVIDIA A100 (sm_80).

Task: row-wise softmax over a row-major float32 matrix x[rows, n]; output y[rows, n]:
    y[r, i] = exp(x[r, i] - max_j x[r, j]) / sum_j exp(x[r, j] - max_j x[r, j])
(The shape contract below calls n "N".)

{contract}

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
- Correct for every shape in the range above (n is often not a multiple of 128) and numerically stable for
  large |x| (x ~ 100).
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


# ---- evaluation (through the common validator interface) ------------------------------------------
def eval_cuda(path: Path) -> dict:
    """VOLTA vs the softmax reference at N = 256, 200, 100 (CPU)."""
    rep = VoltaValidator().validate(Candidate.from_file("softmax", path, entry="softmax_glm"),
                                    ValidationContext(volta_shapes=VOLTA_SHAPES))
    stage = "ok" if rep.passed and rep.decided else (
        "compile" if any(f.kind == "compile_error" for f in rep.findings) else "volta")
    return {"stage": stage, "report": rep}


def eval_tirx(path: Path) -> dict:
    """NumSim (CPU) on 3 tests + Racecheck, through the fixed-test validator with explicit tests."""
    rep = FixedTestsValidator().validate(Candidate.from_file("softmax", path),
                                         ValidationContext(test_cases=TIRX_TESTS, device="cpu"))
    stage = "ok" if rep.passed and rep.decided else (
        "compile" if any(f.kind == "compile_error" for f in rep.findings) else "numsim")
    return {"stage": stage, "report": rep}


def feedback(ev: dict) -> str:
    return render([ev["report"]]) + "\nFix the kernel and return the complete file."


LANGS = {
    "cuda": {"system": CUDA_SYSTEM, "task": cuda_task, "ext": ".cu", "eval": eval_cuda, "feedback": feedback},
    "tirx": {"system": TIRX_SYSTEM, "task": tirx_task, "ext": ".py", "eval": eval_tirx, "feedback": feedback},
}


def main(argv: list[str] | None = None) -> int:
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
    args = ap.parse_args(argv)

    client = LumenClient(model="glm-5.3", per_call_limit=args.per_call_limit, timeout_s=args.http_timeout)
    p_in, p_out = client.price()
    out = RESULTS / "pilot" / datetime.now().strftime("%Y%m%d-%H%M%S")
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
            (out / f"{lang}_r{rnd}_eval.json").write_text(json.dumps({**ev, "report": ev["report"].to_dict() if "report" in ev else None}, indent=1, default=str))
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
    print(f"\nspent {spent:.4f} coins; results in {out}")
    print(json.dumps(summary["langs"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
