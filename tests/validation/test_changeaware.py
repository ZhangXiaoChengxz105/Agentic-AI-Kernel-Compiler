"""Task 2.4: unit tests for changeaware/ (diff, classifier, generator) + a VOLTA integration check.

Run: source env.sh && envs/tirx/bin/python -m unittest tests.test_change_aware -v
"""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.validation.changeaware import plan  # noqa: E402
from akc.validation.changeaware.classify import categories, classify, math_calls  # noqa: E402
from akc.validation.changeaware.diff import diff_sources, normalize  # noqa: E402
from akc.validation.changeaware.generate import extract_thresholds, extract_tiles, generate_tests, volta_shapes  # noqa: E402
from akc.task.workload import Workload, load_workload  # noqa: E402
from akc.validation.volta.layouts import FAIL_STATUSES, make_case  # noqa: E402
from akc.validation.volta.runner import VOLTA_BIN  # noqa: E402

OPT = Path(__file__).resolve().parents[2] / "workloads" / "softmax" / "cuda" / "softmax_opt.cu"
TXL = Path(__file__).resolve().parents[2] / "workloads" / "softmax" / "tirx"
PILOT = Path(__file__).resolve().parents[2] / "results" / "pilot"          # GLM pilot kernels (2026-11-15)


def _bug_pairs():
    for op in ("softmax", "gemm", "layernorm"):
        wl = load_workload(op)
        opt = wl.variant(f"{op}_opt")
        for v in wl.bugs("cuda"):
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


class CosmeticAndRobustnessTests(unittest.TestCase):
    def test_reformat_and_consistent_rename_are_cosmetic(self):
        import re
        src = OPT.read_text()
        self.assertEqual(diff_sources(src, src.replace("; ", ";\n    ")), [])
        self.assertEqual(diff_sources(src, re.sub(r"\bm\b", "row_max", src)), [])

    def test_api_swap_and_partial_rename_are_real_changes(self):
        src = OPT.read_text()
        self.assertTrue(diff_sources(src, src.replace("fmaxf", "fminf")))            # API names may not be renamed
        self.assertTrue(diff_sources(src, src.replace("red[0]", "q[0]")))             # not consistent everywhere

    def test_unparsable_tirx_assumes_every_category(self):
        ref = (TXL / "softmax_txl_ref.py").read_text()
        p = plan("softmax", ref, ref.replace("NEG_BIG", "0.0f", 1), "tirx")
        self.assertEqual(set(p.cats), {"tiling", "indexing", "memory", "sync", "numerical"})


class NewRuleTests(unittest.TestCase):
    """Edits the first rule set missed (notes/build_log.md section 8)."""
    def cats(self, old, new, src=None):
        return categories(classify(diff_sources(old, new), "cuda", src or (old + "\n" + new)))

    def test_reduction_stride_and_start_are_loop_steps(self):
        src = OPT.read_text()
        self.assertIn("indexing", self.cats(src, src.replace("s >>= 1) {\n    if (tid < s) red[tid] +=",
                                                             "s >>= 2) {\n    if (tid < s) red[tid] +=")))
        tail = src[src.index("// 2)"):]
        self.assertIn("indexing", self.cats(src, src[:src.index("// 2)")] + tail.replace("BLOCK / 2", "BLOCK / 4", 1)))

    def test_while_bound_constant(self):
        old = "float l = 0.0f;\nwhile (N - i > 384) {\n  l = l + x;\n}\n"
        self.assertEqual(self.cats(old, old.replace("384", "383")), ["indexing"])

    def test_float_arithmetic_is_numerical_index_arithmetic_is_not(self):
        src = OPT.read_text()
        self.assertEqual(self.cats(src, src.replace("red[tid] += red[tid + s]", "red[tid] = red[tid + s]")), ["numerical"])
        self.assertEqual(self.cats(src, src.replace("* inv;", "* inv * inv;")), ["numerical"])
        self.assertEqual(self.cats(src, src.replace("x + (size_t)blockIdx.x * N", "x + (size_t)blockIdx.x * (N + 1)")),
                         ["indexing"])

    def test_glm_dropped_term_is_numerical(self):
        old = "float l = 0.0f;\nfloat e3;\nl = l + ((e0 + e1) + (e2 + e3));\n"
        self.assertEqual(self.cats(old, old.replace("l = l + ((e0 + e1) + (e2 + e3));", "l = (e0 + e1) + (e2 + l);")),
                         ["numerical"])

    def test_warp_shuffle_is_sync_only_when_the_shuffle_changes(self):
        old = "float m;\nm = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));\nfloat s;\ns += __shfl_xor_sync(0xffffffffu, s, off);\n"
        self.assertIn("sync", self.cats(old, old.replace("__shfl_xor_sync(0xffffffffu, m", "__shfl_down_sync(0xffffffffu, m")))
        self.assertEqual(self.cats(old, old.replace("s += __shfl", "s = __shfl")), ["numerical"])

    def test_tirx_math_function_passed_as_argument(self):
        ref = (TXL / "softmax_txl_ref.py").read_text()
        p = plan("softmax", ref, ref.replace("mx = block_reduce(m, txl.max)", "mx = block_reduce(m, txl.min)"), "tirx")
        self.assertEqual(p.cats, ["numerical"])


class _FakeReply:
    def __init__(self, content):
        self.content, self.cost, self.latency_s, self.model = content, 0.01, 0.1, "fake"
        self.finish_reason, self.dry_run = "stop", False


class _FakeClient:
    def __init__(self, content):
        self.content, self.calls = content, 0

    def chat(self, messages, **kw):
        self.calls += 1
        return _FakeReply(self.content)


class LlmClassifierTests(unittest.TestCase):
    """No network, no coins: a fake client stands in for Lumen."""
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.src = OPT.read_text()
        self.new = self.src.replace("for (int i = tid; i < N; i += BLOCK) sum", "for (int i = tid; i < N - 1; i += BLOCK) sum")

    def tearDown(self):
        self.tmp.cleanup()

    def llm(self, content):
        from akc.validation.changeaware.llm_classify import LlmClassifier
        return LlmClassifier(client=_FakeClient(content), cache_dir=Path(self.tmp.name))

    def test_union_with_rules_and_risky_shapes_become_tests(self):
        reply = ('Sure: {"changes": [{"line": 32, "categories": ["indexing", "numerical"], "why": "sum bound"}], '
                 '"risky_shapes": [{"shape": {"N": 512}, "category": "indexing", "why": "one element past"}, '
                 '{"shape": {"N": 129}, "category": "indexing", "why": "already a floor test"}, '
                 '{"shape": {"N": 99999}, "category": "indexing", "why": "too big"}]}')
        p = plan("softmax", self.src, self.new, "cuda", llm=self.llm(reply))
        self.assertIn("indexing", p.cats)                                    # rules
        self.assertIn("numerical", p.cats)                                   # added by the LLM
        llm_tests = [t for t in p.tests if "LLM-suggested" in t.tag]
        self.assertEqual([t.shape for t in llm_tests], [{"rows": 2, "N": 512}])   # rows defaulted; dup and too-big dropped
        self.assertEqual(p.llm["error"], "")

    def test_garbage_reply_falls_back_to_rules(self):
        rules = plan("softmax", self.src, self.new, "cuda")
        p = plan("softmax", self.src, self.new, "cuda", llm=self.llm("I think it is fine."))
        self.assertEqual(p.cats, rules.cats)
        self.assertIn("no JSON", p.llm["error"])

    def test_cache_and_no_call_for_cosmetic_edits(self):
        reply = '{"changes": [{"line": 32, "categories": ["indexing"], "why": "x"}], "risky_shapes": []}'
        llm = self.llm(reply)
        plan("softmax", self.src, self.new, "cuda", llm=llm)
        p2 = plan("softmax", self.src, self.new, "cuda", llm=llm)
        self.assertEqual(llm.client.calls, 1)
        self.assertTrue(p2.llm["cached"])
        p3 = plan("softmax", self.src, self.src.replace("; ", ";\n"), "cuda", llm=llm)
        self.assertIsNone(p3.llm)                                            # nothing changed: no call
        self.assertEqual(llm.client.calls, 1)


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

    def test_misaligned_pointers_for_cuda_only(self):
        # a compiled tirx-lite function refuses a misaligned tensor before the kernel runs, correct kernel or not
        for op, tiles in (("softmax", (128,)), ("gemm", (16, 16))):
            cuda = generate_tests(op, ["memory"], tiles)
            tirx = generate_tests(op, ["memory"], tiles, lang="tirx")
            self.assertTrue(any(t.misalign for t in cuda), op)
            self.assertFalse(any(t.misalign for t in tirx), op)
            self.assertTrue(any(t.tag.startswith("memory") for t in tirx), op)          # odd sizes stay

    def test_sync_repeats(self):
        sync = [t for t in generate_tests("gemm", ["sync"], (16, 16)) if t.tag.startswith("sync")]
        self.assertTrue(sync and all(t.repeat >= 3 for t in sync))

    def test_deterministic_and_budget(self):
        a = generate_tests("softmax", ["tiling", "numerical", "sync"], (128,), budget=7)
        b = generate_tests("softmax", ["tiling", "numerical", "sync"], (128,), budget=7)
        self.assertEqual(a, b)
        targeted = [t for t in a if not t.tag.startswith("floor")]
        self.assertEqual(len(targeted), 7)                                  # the budget limits targeted tests only
        self.assertEqual({t.tag.split(":")[0] for t in targeted}, {"tiling", "numerical", "sync"})  # round-robin

    def test_floor_runs_even_without_a_category(self):
        for op, tiles in (("gemm", (16, 16)), ("softmax", (128,))):
            tests = generate_tests(op, [], tiles)
            self.assertTrue(tests and all(t.tag.startswith("floor") for t in tests))
            last = "K" if op == "gemm" else "N"
            self.assertTrue(any(t.shape[last] % tiles[0] for t in tests))     # a non-divisible size
            self.assertTrue(any(t.dist != "normal" for t in tests))           # an extreme value distribution

    def test_floor_is_not_duplicated(self):
        keys = [(tuple(sorted(t.shape.items())), t.dist, t.misalign) for t in generate_tests("softmax", ["tiling"], (128,))]
        self.assertEqual(len(keys), len(set(keys)))

    def test_size_regime_switches_are_found(self):
        cuda = ("#define SMEM_FLOATS 8192\n#define CAP (48 * 1024 / sizeof(float))\n"
                "extern \"C\" __global__ void k(const float* x, float* y, int N) {\n"
                "  const bool use_smem = (N <= SMEM_FLOATS);\n  for (int i = tid; i < N; i += 128) {}\n"
                "  for (int k = 0; k < 16; ++k) {}\n  if (N > CAP) return;\n}\n")
        self.assertEqual(extract_thresholds(cuda, "cuda", ["rows", "N"]), [("N", 8192), ("N", 12288)])
        txl = ("def make_kernel(rows: int, n: int):\n    BLOCK = 128\n    NWARP = BLOCK // 32\n"
               "    USE_SMEM = (n * 4 + NWARP * 4) <= (48 * 1024)\n    with txl.if_(col < n), txl.then_():\n        pass\n")
        self.assertEqual(extract_thresholds(txl, "tirx", ["rows", "N"]), [("N", 12284)])
        gemm = (Path(__file__).resolve().parents[2] / "workloads" / "gemm" / "cuda" / "gemm_opt.cu").read_text()
        self.assertEqual(extract_thresholds(gemm, "cuda", ["M", "N", "K"], "tile2d"), [])   # loop var k is not K
        self.assertEqual(extract_thresholds(OPT.read_text(), "cuda", ["rows", "N"]), [])

    def test_regime_tests_run_both_code_paths(self):
        tests = generate_tests("softmax", [], (128,), regimes=[("N", 8192)])
        ns = {t.shape["N"] for t in tests if "switches code path" in t.tag}
        self.assertEqual(ns, {8192, 8193})
        src = OPT.read_text().replace("float m = -INFINITY;", "float m = -INFINITY;\n  const bool big = (N > 9000);")
        p = plan("softmax", OPT.read_text(), src)
        self.assertEqual(p.regimes, [("N", 9000)])
        self.assertTrue({9000, 9001} <= {t.shape["N"] for t in p.tests})

    def test_volta_shapes_put_boundary_in_cta0(self):
        self.assertTrue(any(s["N"] % 128 for s in volta_shapes("softmax", ["tiling"], (128,))))
        self.assertTrue(any(s["M"] < 16 for s in volta_shapes("gemm", ["indexing"], (16, 16))))
        self.assertEqual(volta_shapes("softmax", ["numerical"], (128,)), [{"N": 256}, {"N": 584}])  # default + deep
        self.assertEqual(volta_shapes("layernorm", ["numerical"], (128,)), [{"N": 128}])            # no deep shape
        self.assertTrue(all(s["N"] <= 200 for s in volta_shapes("layernorm", ["sync", "tiling"], (128,))))  # cost cap

    def test_deep_shape_runs_unrolled_paths(self):
        self.assertTrue(any(s["N"] > 4 * 128 for s in volta_shapes("softmax", [], (128,))))


def _no_domain(op: str):
    wl = load_workload(op)
    return Workload(wl.op, wl.dir, {k: v for k, v in wl.meta.items() if k != "domain"}, wl.variants)


def _cap_src(cap: int) -> str:
    return (f"#define CAP {cap}\nextern \"C\" __global__ void k(const float* x, float* y, int N) {{\n"
            "  if (N > CAP) return;\n}\n")


class DomainTests(unittest.TestCase):
    """D1: tests, VOLTA shapes, size regimes and LLM sizes stay inside the manifest's declared shape domain."""
    ALL = ["tiling", "indexing", "memory", "sync", "numerical"]

    def test_generator_never_leaves_the_domain(self):
        for op in ("softmax", "layernorm", "gemm"):
            wl = load_workload(op)
            his = {k: hi for k, (_, hi) in wl.domain.items()}
            regimes = [(k, v) for k, hi in his.items() for v in (hi - 1, hi, 2 * hi)]   # near / at / beyond the max
            for t in (8, 16, 32, 64, 100, 128, 256, 512, 683, 1000, 1024, 2048, 4096):
                for cats in [[c] for c in self.ALL] + [self.ALL, []]:
                    with self.subTest(op=op, tile=t, cats=cats):
                        tests = generate_tests(op, cats, (t,), budget=100, regimes=regimes)
                        self.assertTrue(tests)
                        bad = [x.shape for x in tests if not wl.in_domain(x.shape) or set(x.shape) != set(wl.shape_keys)]
                        self.assertEqual(bad, [])
                        bad = [s for s in volta_shapes(op, cats, (t,)) if not wl.in_domain(s)]
                        self.assertEqual(bad, [])

    def test_clamped_tests_are_tagged_and_not_duplicated(self):
        tests = generate_tests("gemm", ["tiling"], (4096,))
        self.assertTrue(any("clamped into the declared domain" in t.tag for t in tests))
        keys = [(tuple(sorted(t.shape.items())), t.dist, t.misalign) for t in tests]
        self.assertEqual(len(keys), len(set(keys)))

    def test_regime_near_the_max_runs_both_sides_beyond_is_ignored(self):
        def regime_ns(v):
            return {t.shape["N"] for t in generate_tests("softmax", [], (128,), regimes=[("N", v)])
                    if "switches code path" in t.tag}
        self.assertEqual(regime_ns(16383), {16383, 16384})
        self.assertEqual(regime_ns(16384), set())             # N = 16385 is outside: that path never runs
        self.assertEqual(regime_ns(20000), set())

    def test_threshold_beyond_the_domain_is_ignored(self):
        dom = load_workload("softmax").domain
        self.assertEqual(extract_thresholds(_cap_src(16383), "cuda", ["rows", "N"], "row", dom), [("N", 16383)])
        self.assertEqual(extract_thresholds(_cap_src(16384), "cuda", ["rows", "N"], "row", dom), [])
        self.assertEqual(extract_thresholds(_cap_src(20000), "cuda", ["rows", "N"], "row", dom), [])
        self.assertEqual(extract_thresholds(_cap_src(20000), "cuda", ["rows", "N"], "row"), [("N", 20000)])  # REGIME_CAP
        opt = OPT.read_text()
        big = opt.replace("float m = -INFINITY;", "float m = -INFINITY;\n  const bool big = (N > 20000);")
        self.assertEqual(plan("softmax", opt, big).regimes, [])          # plan() uses the manifest's domain

    @unittest.skipUnless(list(PILOT.glob("*/softmax_txl_r*.py")), "needs results/pilot")
    def test_glm_tirx_thresholds_are_inside_the_domain(self):
        dom = load_workload("softmax").domain
        found = []
        for f in sorted(PILOT.glob("*/softmax_txl_r*.py")):
            src = f.read_text()
            with_dom = extract_thresholds(src, "tirx", ["rows", "N"], "row", dom)
            self.assertEqual(with_dom, extract_thresholds(src, "tirx", ["rows", "N"], "row"), f)
            found += [v for _, v in with_dom]
        self.assertTrue(found)
        self.assertTrue(all(12268 <= v and v + 1 <= 16384 for v in found), found)

    def test_llm_sizes_use_the_domain(self):
        import json
        from unittest import mock

        from akc.validation.changeaware import llm_classify as lc
        reply = json.dumps({"changes": [], "risky_shapes": [
            {"shape": {"rows": r, "N": n}, "category": "indexing", "why": "x"}
            for r, n in ((4096, 16384), (300, 100), (2, 16385), (2, 512))]})
        self.assertEqual([s for s, _, _ in lc.parse_reply(reply, "softmax")[1]],
                         [{"rows": 4096, "N": 16384}, {"rows": 300, "N": 100}, {"rows": 2, "N": 512}])
        prompt = lambda: lc.build_messages("softmax", "cuda", "int x;\n", [])[1]["content"]  # noqa: E731
        self.assertIn('"rows" (1..4096), "N" (1..16384)', prompt())
        with mock.patch.object(lc, "load_workload", return_value=_no_domain("softmax")):      # no domain: DIM_CAP
            self.assertEqual([s for s, _, _ in lc.parse_reply(reply, "softmax")[1]], [{"rows": 2, "N": 512}])
            self.assertIn('"rows" (1..256), "N" (1..16384)', prompt())

    def test_without_domain_generator_is_unchanged(self):
        from unittest import mock

        from akc.validation.changeaware import generate as gen
        with mock.patch.object(gen, "load_workload", return_value=_no_domain("gemm")):
            shapes = [t.shape for t in gen.generate_tests("gemm", ["tiling"], (4096,))]
            vshapes = gen.volta_shapes("gemm", ["tiling"], (4096,))
        self.assertIn({"M": 24580, "N": 24580, "K": 8200}, shapes)     # sizes are not limited without a domain
        self.assertIn({"M": 8192, "N": 8192, "K": 8200}, vshapes)
        with mock.patch.object(gen, "load_workload", return_value=_no_domain("softmax")):
            ns = {t.shape["N"] for t in gen.generate_tests("softmax", [], (128,), regimes=[("N", 20000)])}
        self.assertTrue({20000, 20001} <= ns)


@unittest.skipUnless(shutil.which("nvcc") and VOLTA_BIN.exists(), "needs nvcc and VOLTA")
class VoltaIntegration(unittest.TestCase):
    def test_change_aware_shapes_expose_shape_dependent_bugs(self):
        from akc.backend.ptx import compile_to_ptx
        from akc.validation.volta.runner import run_compare
        out = Path(__file__).resolve().parents[2] / "build" / "ptx_ca_test"
        for op, name in (("softmax", "softmax_bug_tail"), ("gemm", "gemm_bug_guard")):
            wl = load_workload(op)
            ref, opt, bug = wl.reference("cuda"), wl.variant(f"{op}_opt"), wl.variant(name)
            p = plan(op, opt.file.read_text(), bug.file.read_text())
            rp, bp = (compile_to_ptx(x.file, out / op / x.name).ptx for x in (ref, bug))
            statuses = [run_compare(make_case(op, rp, ref, bp, bug, s)).status for s in p.volta]
            self.assertEqual(statuses[0], "EQUIV", f"{name}: default shape should miss it")   # statuses[1] = deep shape
            self.assertTrue(set(statuses) & set(FAIL_STATUSES), f"{name}: {statuses}")


if __name__ == "__main__":
    unittest.main()


# ---- tirx-lite (Python) support ------------------------------------------------------------------
TXL = Path(__file__).resolve().parents[2] / "workloads" / "softmax" / "tirx"
RMSNORM = Path(__file__).resolve().parents[2] / "third_party" / "tirx-kernels" / "tirx_kernels" / "norm" / "rmsnorm.py"


class TirxTests(unittest.TestCase):
    def test_language_detection(self):
        from akc.validation.changeaware.diff import detect_lang
        self.assertEqual(detect_lang((TXL / "softmax_txl_ref.py").read_text()), "tirx")
        self.assertEqual(detect_lang(OPT.read_text()), "cuda")

    def test_txl_bug_variants_get_their_category_first(self):
        wl = load_workload("softmax")
        ref = wl.reference("tirx").file.read_text()
        for v in wl.bugs("tirx"):
            with self.subTest(bug=v.name):
                cats = categories(classify(diff_sources(ref, v.file.read_text()), "tirx"))
                self.assertEqual(cats[:1], [v.bug["category"]], cats)

    def test_shape_list_is_not_an_index_expression(self):
        old = 'red = smem.alloc([NWARP], "float32")'
        new = 'red = smem.alloc([NWARP * 2], "float32")'
        cats = categories(classify(diff_sources(f"import tirx_kernels\n{old}\n", f"import tirx_kernels\n{new}\n"), "tirx"))
        self.assertIn("memory", cats)
        self.assertNotIn("indexing", cats)          # [NWARP] is a shape list, not a subscript

    def test_cosmetic_python_changes_ignored(self):
        src = (TXL / "softmax_txl_ref.py").read_text()
        cosmetic = src.replace("        r = txl.cta_id()\n", "        # which row this CTA owns\n        r   =   txl.cta_id( )\n")
        self.assertEqual(diff_sources(src, cosmetic), [])

    @unittest.skipUnless(RMSNORM.exists(), "needs third_party/tirx-kernels")
    def test_official_rmsnorm_mutations(self):
        src = RMSNORM.read_text()
        cases = [
            ("            cta_sync(0)\n            with txl.if_(warp == 0):", "            with txl.if_(warp == 0):", "sync"),
            ("txl.if_(st < hidden_size)", "txl.if_(st <= hidden_size)", "indexing"),
            ("block = min(256, hidden_size // vec)", "block = min(128, hidden_size // vec)", "tiling"),
            (" + txl.float32(eps)))", "))", "numerical"),
            ('sum_sq_smem = smem.alloc([bdy], "float32")', 'sum_sq_smem = smem.alloc([bdy * 2], "float32")', "memory"),
        ]
        for old, new, want in cases:
            with self.subTest(want=want):
                self.assertIn(old, src)
                cats = categories(classify(diff_sources(src, src.replace(old, new, 1)), "tirx"))
                self.assertEqual(cats[:1], [want], cats)

    def test_plan_autodetects_tirx(self):
        ref = (TXL / "softmax_txl_ref.py").read_text()
        p = plan("softmax", ref, (TXL / "bugs" / "softmax_txl_bug_tail.py").read_text())
        self.assertEqual(p.cats[0], "tiling")
        self.assertTrue(any(t.shape["N"] % 128 for t in p.tests))
