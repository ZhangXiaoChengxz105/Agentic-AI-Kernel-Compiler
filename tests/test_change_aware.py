"""Task 2.4: unit tests for changeaware/ (diff, classifier, generator) + a VOLTA integration check.

Run: source env.sh && envs/tirx/bin/python -m unittest tests.test_change_aware -v
"""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from changeaware import plan  # noqa: E402
from changeaware.classify import categories, classify, math_calls  # noqa: E402
from changeaware.diff import diff_sources, normalize  # noqa: E402
from changeaware.generate import generate_tests, volta_shapes  # noqa: E402
from verify.cases import FAIL_STATUSES, load_manifest, make_case  # noqa: E402
from verify.volta import VOLTA_BIN  # noqa: E402

OPT = Path(__file__).resolve().parents[1] / "kernels" / "softmax" / "softmax_opt.cu"


def _bug_pairs():
    for op in ("softmax", "gemm", "layernorm"):
        _, vs = load_manifest(op)
        opt = next(v for v in vs if v.name == f"{op}_opt")
        for v in vs:
            if v.bug:
                yield op, opt, v


class DiffTests(unittest.TestCase):
    def test_comments_whitespace_and_kernel_name_ignored(self):
        src = OPT.read_text()
        cosmetic = (src.replace("// 1) row max", "// first: the row maximum")
                       .replace("softmax_opt(", "renamed_kernel(").replace("  m = red[0];", "  m   =   red[0];"))
        self.assertEqual(diff_sources(src, cosmetic), [])

    def test_line_numbers_survive_block_comments(self):
        lines = normalize("/* a\n b\n c */\nint x;\n")
        self.assertEqual(lines, [(4, "int x;")])

    def test_detects_deleted_line(self):
        src = OPT.read_text()
        h = diff_sources(src, src.replace("  m = red[0];\n", "", 1))
        self.assertEqual([x.kind for x in h], ["del"])


class ClassifyTests(unittest.TestCase):
    def test_every_injected_bug_gets_its_category_first(self):
        for op, opt, bug in _bug_pairs():
            with self.subTest(bug=bug.name):
                cats = categories(classify(diff_sources(opt.file.read_text(), bug.file.read_text())))
                self.assertTrue(cats, "no category")
                self.assertEqual(cats[0], bug.bug["category"], cats)

    def test_math_rule_ignores_unchanged_math_call_on_changed_line(self):
        old = "for (int i = tid; i < N; i += BLOCK) m = fmaxf(m, xr[i]);"
        new = "for (int i = tid; i < 2 * N; i += BLOCK) m = fmaxf(m, xr[i]);"
        self.assertNotIn("numerical", categories(classify(diff_sources(old, new))))

    def test_math_calls_extracts_balanced_arguments(self):
        self.assertEqual(math_calls("y = EXP(x[i] - fmaxf(a, b)) * inv;"), ["EXP(x[i] - fmaxf(a, b))", "fmaxf(a, b)"])

    def test_added_barrier_is_sync_and_new_shared_buffer_is_memory(self):
        cats = categories(classify(diff_sources("a = 1;\n", "__shared__ float t[64];\na = 1;\n__syncthreads();\n")))
        self.assertEqual(set(cats), {"memory", "sync"})


class GenerateTests(unittest.TestCase):
    def test_tiling_row_op_covers_partial_and_sub_block_sizes(self):
        ns = {t.shape["N"] for t in generate_tests("softmax", ["tiling"], (128,))}
        self.assertTrue(any(n % 128 for n in ns))
        self.assertTrue(any(n < 128 for n in ns))

    def test_tiling_gemm_covers_k_tail_and_sub_tile_matrix(self):
        shapes = [t.shape for t in generate_tests("gemm", ["tiling"], (16, 16))]
        self.assertTrue(any(s["K"] % 16 for s in shapes))
        self.assertTrue(any(s["M"] < 16 and s["N"] < 16 for s in shapes))

    def test_numerical_uses_stress_distributions(self):
        self.assertIn("large", {t.dist for t in generate_tests("softmax", ["numerical"], (128,))})
        self.assertIn("offset", {t.dist for t in generate_tests("layernorm", ["numerical"], (128,))})

    def test_sync_repeats(self):
        self.assertTrue(all(t.repeat >= 3 for t in generate_tests("gemm", ["sync"], (16, 16))))

    def test_deterministic_and_budget(self):
        a = generate_tests("softmax", ["tiling", "numerical", "sync"], (128,), budget=7)
        b = generate_tests("softmax", ["tiling", "numerical", "sync"], (128,), budget=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 7)
        self.assertEqual({t.tag.split(":")[0] for t in a}, {"tiling", "numerical", "sync"})  # round-robin

    def test_no_category_gives_sanity_tests(self):
        self.assertEqual(len(generate_tests("gemm", [], (16, 16))), 2)

    def test_volta_shapes_put_boundary_in_cta0(self):
        self.assertTrue(any(s["N"] % 128 for s in volta_shapes("softmax", ["tiling"], (128,))))
        self.assertTrue(any(s["M"] < 16 for s in volta_shapes("gemm", ["indexing"], (16, 16))))
        self.assertEqual(volta_shapes("softmax", ["numerical"], (128,)), [{"N": 256}])   # VOLTA can't help


@unittest.skipUnless(shutil.which("nvcc") and VOLTA_BIN.exists(), "needs nvcc and VOLTA")
class VoltaIntegration(unittest.TestCase):
    def test_change_aware_shapes_expose_shape_dependent_bugs(self):
        from verify.ptx import compile_to_ptx
        out = Path(__file__).resolve().parents[1] / "build" / "ptx_ca_test"
        for op, name in (("softmax", "softmax_bug_tail"), ("gemm", "gemm_bug_guard")):
            _, vs = load_manifest(op)
            v = {x.name: x for x in vs}
            ref, opt, bug = v[f"{op}_ref"], v[f"{op}_opt"], v[name]
            p = plan(op, opt.file.read_text(), bug.file.read_text(), bug.block)
            rp, bp = (compile_to_ptx(x.file, out / op).ptx for x in (ref, bug))
            from verify.volta import run_compare
            statuses = [run_compare(make_case(op, rp, ref, bp, bug, s)).status for s in p.volta]
            self.assertEqual(statuses[0], "EQUIV", f"{name}: default shape should miss it")
            self.assertTrue(set(statuses) & set(FAIL_STATUSES), f"{name}: {statuses}")


if __name__ == "__main__":
    unittest.main()
