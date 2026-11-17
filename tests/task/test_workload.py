"""Workload simplification: optional manifest / OpSpec parts, derived launch and derived VOLTA layout.

Run: source env.sh && python -m unittest tests.task.test_workload -v
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.task.reference import OPS  # noqa: E402
from akc.task.workload import DEFAULT_DISTS, Workload, list_workloads, load_workload  # noqa: E402
from akc.validation.interface import Candidate, ValidationContext  # noqa: E402
from akc.validation.volta.layouts import derive_layout  # noqa: E402

ROWS = 2


def _cdiv(a, b):
    return -(-a // b)


# The hand-written layouts used until 2026-11-16; the derived ones must reproduce them exactly.
HAND_WRITTEN = {
    "softmax": (lambda s: [("x", 4, ROWS * s["N"], "in"), ("y", 4, ROWS * s["N"], "out")],
                lambda s: ["ptr:x", "ptr:y", f"int:{s['N']}"], lambda s: (ROWS,), ("y",)),
    "layernorm": (lambda s: [("x", 4, ROWS * s["N"], "in"), ("w", 4, s["N"], "in"), ("b", 4, s["N"], "in"),
                             ("y", 4, ROWS * s["N"], "out")],
                  lambda s: ["ptr:x", "ptr:w", "ptr:b", "ptr:y", f"int:{s['N']}", "float:1e-05"], lambda s: (ROWS,), ("y",)),
    "gemm": (lambda s: [("A", 4, s["M"] * s["K"], "in"), ("B", 4, s["K"] * s["N"], "in"), ("C", 4, s["M"] * s["N"], "out")],
             lambda s: ["ptr:A", "ptr:B", "ptr:C", f"int:{s['M']}", f"int:{s['N']}", f"int:{s['K']}"],
             lambda s: (_cdiv(s["N"], 16), _cdiv(s["M"], 16)), ("C",)),
}
SHAPES = {"softmax": [{"N": 256}, {"N": 200}, {"N": 584}], "layernorm": [{"N": 128}, {"N": 100}],
          "gemm": [{"M": 32, "N": 32, "K": 48}, {"M": 12, "N": 12, "K": 40}, {"M": 16, "N": 16, "K": 64}]}


def _stripped(op: str, drop: tuple[str, ...]) -> Workload:
    wl = load_workload(op)
    return Workload(wl.op, wl.dir, {k: v for k, v in wl.meta.items() if k not in drop}, wl.variants)


class DerivedVoltaLayoutTests(unittest.TestCase):
    def test_matches_the_hand_written_layouts(self):
        for op, (arrays, params, grid, check) in HAND_WRITTEN.items():
            lay = derive_layout(op)
            for s in SHAPES[op]:
                with self.subTest(op=op, shape=s):
                    self.assertEqual(lay.arrays(s), arrays(s))
                    self.assertEqual(lay.params(s), params(s))
                    self.assertEqual(lay.grid(s), grid(s))
                    self.assertEqual(lay.check, check)


class OptionalPartsTests(unittest.TestCase):
    def test_defaults_for_missing_optional_keys(self):
        wl = _stripped("softmax", ("tiles", "numerical_dists", "perf", "volta", "references"))
        self.assertEqual(wl.tiles, [128])                       # the launch block
        self.assertEqual(wl.numerical_dists, list(DEFAULT_DISTS))
        self.assertIsNone(wl.perf_shape)
        self.assertIsNone(wl.volta)
        self.assertFalse(wl.has_reference("cuda"))
        self.assertEqual(_stripped("gemm", ("tiles",)).tiles, [16])   # the launch tile

    def test_launch_is_derived_from_the_manifest(self):
        self.assertIsNone(OPS["softmax"].launch)
        self.assertEqual(load_workload("gemm").launch_dims({"M": 33, "N": 17, "K": 8}, (16, 16)), ((2, 3, 1), (16, 16, 1)))

    def test_volta_not_set_up_is_info_not_a_crash(self):
        from akc.validation.volta import validator as vmod
        wl = load_workload("softmax")
        cand = Candidate.from_file("softmax", wl.variant("softmax_opt").file, entry="softmax_opt")
        with mock.patch.object(vmod, "load_workload", return_value=_stripped("softmax", ("volta",))):
            rep = vmod.VoltaValidator().validate(cand, ValidationContext(device="cpu"))
        self.assertTrue(rep.passed)
        self.assertFalse(rep.decided)
        self.assertIn("not set up", rep.findings[0].message)

    def test_speed_not_set_up(self):
        from akc.perf import measure
        from akc.task import reference
        wl = load_workload("softmax")
        cand = Candidate.from_file("softmax", wl.variant("softmax_opt").file, entry="softmax_opt")
        with mock.patch.dict(reference.OPS, {"softmax": replace(OPS["softmax"], flops=None)}):
            r = measure(cand, device="gpu")
        self.assertFalse(r.measured)
        self.assertIn("not set up", r.reason)


class DomainTests(unittest.TestCase):
    """D1: the declared shape range (CAKE: "the valid shape domain is declared before tuning") + one benchmark shape."""
    RANGES = {"softmax": {"rows": (1, 4096), "N": (1, 16384)}, "layernorm": {"rows": (1, 4096), "N": (1, 16384)},
              "gemm": {"M": (1, 4096), "N": (1, 4096), "K": (1, 4096)}}

    def test_declared_ranges(self):
        for op, want in self.RANGES.items():
            self.assertEqual(load_workload(op).domain, want, op)

    def test_manifest_shapes_are_inside_the_domain(self):
        from akc.validation.changeaware.generate import volta_shapes
        from akc.validation.fixed_tests.suite import fixed_suite
        for op in list_workloads():
            wl = load_workload(op)
            self.assertIsNotNone(wl.domain, op)
            v = wl.volta or {}
            shapes = ([wl.perf_shape] + [t.shape for t in fixed_suite(op)] + list(wl.meta["fixed_suite"])
                      + [s for s in (v.get("default"), v.get("deep")) if s]
                      + volta_shapes(op, ["tiling", "indexing", "memory", "sync", "numerical"]))
            for s in shapes:
                with self.subTest(op=op, shape=s):
                    self.assertTrue(wl.in_domain(s))

    def test_in_domain_and_clip(self):
        wl = load_workload("softmax")
        self.assertTrue(wl.in_domain({"rows": 4096, "N": 16384}))
        self.assertTrue(wl.in_domain({"rows": 1, "N": 1}))
        self.assertTrue(wl.in_domain({"N": 256}))                       # partial shape (VOLTA): checked on N only
        for bad in ({"rows": 4097, "N": 1}, {"rows": 2, "N": 16385}, {"rows": 2, "N": 0}):
            self.assertFalse(wl.in_domain(bad), bad)
        self.assertEqual(wl.clip({"rows": 5000, "N": 0}), {"rows": 4096, "N": 1})
        self.assertEqual(wl.clip({"N": 300}), {"N": 300})
        self.assertEqual(wl.domain_text(), "rows in [1, 4096], N in [1, 16384]")

    def test_bad_domain_is_rejected(self):
        wl = load_workload("softmax")
        for d in ({"rows": [10, 1]}, {"rows": [0, 4]}, {"B": [1, 2]}):
            with self.subTest(domain=d), self.assertRaises(ValueError):
                Workload(wl.op, wl.dir, {**wl.meta, "domain": d}, wl.variants).domain

    def test_without_domain_nothing_is_limited(self):
        wl = _stripped("gemm", ("domain",))
        self.assertIsNone(wl.domain)
        self.assertTrue(wl.in_domain({"M": 10 ** 6, "N": 1, "K": 1}))
        self.assertEqual(wl.clip({"M": 10 ** 6, "N": 0}), {"M": 10 ** 6, "N": 0})
        self.assertEqual(wl.domain_text(), "")

    def test_contract_text_states_range_and_benchmark_shape(self):
        from akc.agent.pilot import contract_text
        for op, rng, bench in (("softmax", "rows in [1, 4096], N in [1, 16384]", "rows=4096, N=1024"),
                               ("gemm", "M in [1, 4096], N in [1, 4096], K in [1, 4096]", "M=1024, N=1024, K=1024")):
            for lang in ("tirx", "cuda"):
                with self.subTest(op=op, lang=lang):
                    text = contract_text(load_workload(op), lang)
                    self.assertIn(rng, text)
                    self.assertIn(f"Speed is measured at {bench}", text)
        self.assertIn("traced once per shape", contract_text(load_workload("softmax"), "tirx"))
        self.assertIn("no shape range is declared", contract_text(_stripped("softmax", ("domain",)), "tirx"))

    def test_pilot_prompts_carry_the_contract(self):
        from akc.agent import pilot
        self.assertIn("N in [1, 16384]", pilot.cuda_task())                  # string building only: no Lumen call
        if (pilot.TXL_DIR / "tirx_lite" / "README.md").exists():
            self.assertIn("Speed is measured at rows=4096, N=1024", pilot.tirx_task())


if __name__ == "__main__":
    unittest.main()
