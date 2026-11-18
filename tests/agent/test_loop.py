"""Tests for the agent repair loop (akc/agent: prompts, context, loop, trajectory).

No model calls and no GPU: a fake LLM returns scripted replies, and run_pipeline / perf.measure are replaced by
fakes that decide pass / fail from a marker in the code. The real pipeline is covered by tests/validation.

Run: source env.sh && python -m unittest tests.agent.test_loop -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import akc.perf  # noqa: E402
from akc.agent import loop  # noqa: E402
from akc.agent.prompts import extract_code, task_prompt  # noqa: E402
from akc.llm.lumen import ChatResult, LumenError  # noqa: E402
from akc.perf.hardware import spec_for  # noqa: E402
from akc.perf.report import analyze  # noqa: E402
from akc.validation.interface import Finding, ValidationReport  # noqa: E402
from akc.validation.pipeline import PipelineResult  # noqa: E402


def code(tag: str) -> str:
    return f"```python\nimport tirx_kernels.tirx_lite as txl\n# {tag}\n```"


class FakeLLM:
    model = "fake"

    def __init__(self, replies, price=(0.0, 0.0), fail_at=None):
        self.replies, self._price, self.fail_at, self.calls = list(replies), price, fail_at, []

    def price(self, model=None):
        return self._price

    @staticmethod
    def estimate_tokens(messages):
        return sum(len(m["content"]) for m in messages) // 4

    def chat(self, messages, max_tokens=0, dry_run=False, tag="", **extra):
        self.calls.append(messages)
        if self.fail_at == len(self.calls):
            raise LumenError("HTTP 503")
        if dry_run:
            return ChatResult(self.model, "[dry-run]", dry_run=True)
        return ChatResult(self.model, self.replies.pop(0), input_tokens=10, output_tokens=20, cost=0.01,
                          latency_s=0.1, finish_reason="stop")


class Patched:
    """Replace run_pipeline and perf.measure: '# PASS' / '# PASS 0.5' (speed in ms) pass, anything else fails."""

    def __init__(self):
        self.cands = []

    def pipeline(self, cand, config, ctx):
        self.cands.append(cand)
        res = PipelineResult(config)
        if "# PASS" in cand.source:
            res.reports = [ValidationReport("fixed_tests", True, True)]
        else:
            res.reports = [ValidationReport("fixed_tests", True, True),
                           ValidationReport("volta", False, True, [Finding("race", "error", "volta", "race on smem")])]
            res.stopped_after = "volta"
        return res

    def measure(self, cand, device="auto"):
        t = float(cand.source.split("# PASS")[1].split()[0]) if "# PASS " in cand.source else 1.0
        return analyze(cand.workload, {"rows": 4096, "N": 1024}, {"median_ms": t, "iters": 50},
                       spec_for("A100-SXM4-40GB"), label=cand.label, lang="tirx", torch_ms=1.0, copy_ms=0.3)

    def __enter__(self):
        self._p = [mock.patch.object(loop, "run_pipeline", self.pipeline),
                   mock.patch.object(akc.perf, "measure", self.measure)]
        for p in self._p:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._p:
            p.stop()


class TestPrompts(unittest.TestCase):
    def test_task_prompt_every_workload(self):
        for w in ("softmax", "layernorm", "gemm"):
            t = task_prompt(w)
            self.assertIn("make_kernel(", t)
            self.assertIn("Shape contract", t)
            self.assertIn("sm_80", t)
            self.assertIn("```python", t)

    def test_extract_code(self):
        self.assertEqual(extract_code("text\n```python\nx = 1\n```\n"), "x = 1\n")
        self.assertIsNone(extract_code("no code here"))


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def run_agent(self, llm, **kw):
        return loop.run_agent("softmax", "C", llm=llm, out_dir=self.out, **kw)

    def test_repair_then_stop_at_first_pass(self):
        llm = FakeLLM([code("bug"), code("PASS"), code("never asked")])
        with Patched() as p:
            r = self.run_agent(llm)
        self.assertTrue(r.passed)
        self.assertEqual(r.stop_reason, "passed")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(r.best.label, "it2")
        self.assertIsNone(p.cands[1].parent)                    # it1 failed: still compared with the reference
        second = llm.calls[1][-1]["content"]                   # the observation of it1 reached the model
        self.assertIn("race on smem", second)
        self.assertIn("# bug", second)
        self.assertIn("it1: FAILED (volta)", second)
        self.assertEqual(len(llm.calls[0]), 2)                 # first turn: system + task only

    def test_cap_at_max_iters(self):
        llm = FakeLLM([code(f"bug{i}") for i in range(10)])
        with Patched():
            r = self.run_agent(llm)
        self.assertFalse(r.passed)
        self.assertEqual((r.stop_reason, len(llm.calls), len(r.steps)), ("max_iters", 10, 10))
        lines = (self.out / "trajectory.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 10)
        self.assertEqual(json.loads(lines[0])["failed_validators"], ["volta"])
        for f in ("run.json", "summary.json", "it1_prompt.json", "it1_reply.md", "it1.py", "it1_validation.json",
                  "it10_observation.txt"):
            self.assertTrue((self.out / f).exists(), f)
        self.assertFalse((self.out / "best.py").exists())

    def test_optimize_keeps_fastest_and_parent_is_last_accepted(self):
        llm = FakeLLM([code("PASS 0.5"), code("bug"), code("PASS 0.4"), code("PASS 0.9")])
        with Patched() as p:
            r = self.run_agent(llm, optimize=True, max_iters=4)
        self.assertEqual(r.stop_reason, "max_iters")
        self.assertEqual(r.best.label, "it3")
        self.assertEqual(p.cands[2].parent.label, "it1")          # it2 failed, so it3 is compared with it1
        self.assertEqual(p.cands[3].parent.label, "it3")
        self.assertIn("make it faster", llm.calls[1][-1]["content"].lower())
        self.assertIn("Best correct version so far (it1, 0.5000 ms", llm.calls[2][-1]["content"])
        self.assertIn("# PASS 0.4", (self.out / "best.py").read_text())

    def test_no_code_reply(self):
        llm = FakeLLM(["I think the answer is ...", code("PASS")])
        with Patched():
            r = self.run_agent(llm)
        self.assertEqual([s.code is None for s in r.steps], [True, False])
        self.assertIn("no fenced code block", llm.calls[1][-1]["content"])

    def test_seed_is_iteration_zero(self):
        seed = Path(self.tmp.name) / "seed.py"
        seed.write_text("# bug in seed\n")
        llm = FakeLLM([code("PASS")])
        with Patched():
            r = self.run_agent(llm, seed=seed)
        self.assertEqual([s.label for s in r.steps], ["seed", "it1"])
        self.assertIsNone(r.steps[0].llm)
        self.assertIn("# bug in seed", llm.calls[0][-1]["content"])
        passing = Path(self.tmp.name) / "ok.py"
        passing.write_text("# PASS\n")
        llm2 = FakeLLM([])
        with Patched():
            r2 = loop.run_agent("softmax", "A", llm=llm2, seed=passing, out_dir=Path(self.tmp.name) / "run2")
        self.assertEqual((r2.stop_reason, len(llm2.calls)), ("passed", 0))

    def test_budget_and_llm_error_stop(self):
        with Patched():
            r = self.run_agent(FakeLLM([code("bug")], price=(1.0, 100.0)), max_coins=1.0, max_tokens=100000)
        self.assertEqual((r.stop_reason, len(r.steps)), ("budget", 0))
        with Patched():
            r = loop.run_agent("softmax", "C", llm=FakeLLM([code("bug")], fail_at=2), out_dir=self.out / "e")
        self.assertEqual(r.stop_reason, "llm_error")
        self.assertIn("HTTP 503", r.steps[-1].observation)

    def test_dry_run(self):
        llm = FakeLLM([])
        with Patched():
            r = self.run_agent(llm, dry_run=True)
        self.assertEqual(r.stop_reason, "dry_run")
        self.assertTrue((self.out / "it1_prompt.json").exists())

    def test_unknown_config(self):
        with self.assertRaises(ValueError):
            loop.run_agent("softmax", "D", llm=FakeLLM([]), out_dir=self.out)


if __name__ == "__main__":
    unittest.main()
