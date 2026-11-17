"""Task 2 (D3): IR change analysis for tirx-lite (irdiff, classify_ir, ir_worker, plan's IR path).

Tracing needs tvm_tirx: those tests are skipped without it (the pure-Python classifier tests always run).
All revisions are traced in ONE child process (ir_worker.prefetch in setUpClass): tvm is imported once.
Run: source env.sh && python -m unittest tests.validation.test_changeaware_ir -v
"""
from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.validation.changeaware import PROPOSAL_CATEGORIES, generate_tests, plan, proposal_category  # noqa: E402
from akc.validation.changeaware.classify import CATEGORIES, categories  # noqa: E402
from akc.validation.changeaware.classify_ir import classify_ir, op_class  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TXL = ROOT / "workloads" / "softmax" / "tirx"
REF = (TXL / "softmax_txl_ref.py").read_text()
BUG = {b: (TXL / "bugs" / f"softmax_txl_bug_{b}.py").read_text() for b in ("offbyone", "tail", "nosync", "nomax")}
D1 = ROOT / "results" / "pilot" / "20261115-204306" / "softmax_txl_r2.py"
HAS_TVM = importlib.util.find_spec("tvm_tirx") is not None

SWAP = REF.replace("txl.assign(s, s + txl.exp(v - mx))", "txl.assign(s, s + txl.exp(mx - v))")
BUFFER = REF.replace("txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))\n                txl.assign(m, txl.max(m, v))",
                     "txl.ptx.ld.global_.f32(v, y.ptr_to([r, col]))\n                txl.assign(m, txl.max(m, v))")
RENAME = re.sub(r"\bm\b", "row_max", REF)                   # a local renamed everywhere: no IR change
RETRACE = REF + "\n# same kernel, one more comment\n"


def _d1_float16():
    """glm_txl_D1_r2 with `m = txl.local_scalar("float32")` made float16: the trace itself fails."""
    src = D1.read_text()
    return src.replace('m = txl.local_scalar("float32")', 'm = txl.local_scalar("float16")', 1)


@unittest.skipUnless(HAS_TVM, "needs tvm_tirx (tirx-lite tracing)")
class IrPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from akc.validation.changeaware import ir_worker
        news = list(BUG.values()) + [SWAP, BUFFER, RENAME, RETRACE]
        items = [(REF, s, ir_worker.trace_shapes("softmax", REF, s)) for s in news]
        if D1.exists():
            items.append((D1.read_text(), _d1_float16(), ir_worker.trace_shapes("softmax", D1.read_text(), _d1_float16())))
        ir_worker.prefetch("softmax", items)

    def plan(self, new, old=REF):
        return plan("softmax", old, new, "tirx")

    def test_offbyone_is_indexing_at_the_guard_line(self):
        p = self.plan(BUG["offbyone"])
        self.assertEqual(p.analysis["mode"], "ir")
        self.assertEqual(p.cats[0], "indexing")
        self.assertIn(("indexing", "guard", 51), {(lab.category, lab.rule, lab.line) for lab in p.labels})
        self.assertEqual(len(p.hunks), 1)
        self.assertEqual((p.hunks[0].old[0][0], p.hunks[0].new[0][0]), (51, 54))   # old / new source lines
        self.assertIn("LT -> LE", next(lab.evidence for lab in p.labels if lab.rule == "guard"))

    def test_tail_is_tiling_and_only_visible_at_non_multiples(self):
        p = self.plan(BUG["tail"])
        self.assertEqual(p.cats[0], "tiling")
        self.assertEqual({lab.line for lab in p.labels if lab.rule == "trip-count"}, {41, 49})
        per_shape = p.analysis["hunks"]["ir_per_shape"]
        self.assertEqual(per_shape["rows=2,N=1024"], 0)              # benchmark shape: identical IR
        self.assertGreater(per_shape["rows=2,N=129"], 0)
        self.assertTrue(all("not at rows=2,N=1024" in h.old[0][1] for h in p.hunks))

    def test_nosync_collapses_to_one_sync_hunk(self):
        p = self.plan(BUG["nosync"])
        self.assertEqual(p.cats, ["sync"])
        self.assertEqual(len(p.hunks), 1)
        self.assertEqual(p.hunks[0].kind, "del")
        self.assertIn("×14", p.hunks[0].old[0][1])                  # 7 unrolled steps x 2 block_reduce calls
        self.assertEqual(p.hunks[0].old[0][0], 34)
        self.assertTrue(all(n == 14 for n in p.analysis["hunks"]["ir_per_shape"].values()))

    def test_nomax_is_numerical(self):
        p = self.plan(BUG["nomax"])
        self.assertEqual(p.cats, ["numerical"])
        self.assertEqual({lab.line for lab in p.labels}, {53, 60})

    def test_retrace_and_rename_give_no_hunks(self):
        for new in (RETRACE, RENAME):
            p = self.plan(new)
            self.assertEqual(p.analysis["mode"], "ir")
            self.assertEqual((p.hunks, p.cats), ([], []))
            self.assertFalse(p.analysis["ir_blind"])

    def test_swapped_operands_and_other_buffer_are_detected(self):
        p = self.plan(SWAP)
        self.assertEqual(p.cats, ["numerical"])
        self.assertEqual(len(p.hunks), 1)
        p = self.plan(BUFFER)
        self.assertEqual(len(p.hunks), 1)
        self.assertIn(("memory", "buffer"), {(lab.category, lab.rule) for lab in p.labels})

    @unittest.skipUnless(D1.exists(), "needs results/pilot")
    def test_float16_mutant_that_does_not_trace_falls_back_to_the_ast(self):
        p = plan("softmax", D1.read_text(), _d1_float16(), "tirx")
        self.assertEqual(p.analysis["mode"], "ast-fallback")
        self.assertIn("does not trace", p.analysis["fallback_reason"])
        self.assertIn("numerical", p.cats)                            # the AST labels are used

    def test_same_categories_give_the_same_tests(self):
        ir = self.plan(BUG["nosync"])
        ast = plan("softmax", REF, BUG["nosync"], "tirx", classifier="ast")
        self.assertEqual(ir.cats, ast.cats)
        self.assertEqual([t.key() for t in ir.tests], [t.key() for t in ast.tests])
        self.assertEqual(ast.analysis["mode"], "ast-fallback")


class ClassifyIrTests(unittest.TestCase):
    """classify_ir works on JSON hunks: no tvm needed."""
    @staticmethod
    def rec(kind, text, line, reduce=False, op=None):
        feats = {"reduce": reduce, "ops": [op] if op else []}
        if op:
            feats["op"] = op
        return {"kind": kind, "text": text, "line": line, "feats": feats, "depth": 2}

    def hunk(self, tag, old, new, changes=()):
        pairs = [[0, 0]] if tag == "mod" else []
        return {"tag": tag, "old": old, "new": new, "pairs": pairs, "count": 1, "shapes": ["N=1"],
                "changes": [{**c, "pair": [0, 0]} for c in changes]}

    def test_categories_follow_the_changed_ir_node(self):
        r = self.rec
        cases = [
            (self.hunk("mod", [r("For", "for v in range(2):", 10)], [r("For", "for v in range(1):", 11)],
                       [{"slot": "end", "path": "end.value", "old_node": "int", "new_node": "int"}]), {"tiling"}),
            (self.hunk("mod", [r("If", "if a < n:", 12)], [r("If", "if a <= n:", 12)],
                       [{"slot": "condition", "path": "condition", "old_node": "LT", "new_node": "LE"}]), {"indexing"}),
            (self.hunk("del", [r("Evaluate", "T.cuda.cta_sync()", 20, op="cuda.cta_sync")], []), {"sync"}),
            (self.hunk("mod", [r("Evaluate", "T.ptx.ld(...)", 30, op="ptx.ld")], [r("Evaluate", "T.ptx.ld(...)", 30, op="ptx.ld")],
                       [{"slot": "value.args[12]", "path": "value.args[12]", "old_node": "StringImm",
                         "new_node": "StringImm", "old_text": "''", "new_text": "'v4'", "calls": ["ptx.ld"]}]), {"memory"}),
            (self.hunk("mod", [r("TensorStore", "v[0] = v[0] + T.exp(a - b)", 40)], [r("TensorStore", "v[0] = ...", 40)],
                       [{"slot": "value", "path": "value.b.args[0]", "old_node": "Sub", "new_node": "TensorLoad",
                         "dtype": "float32", "calls": ["prim.exp"]}]), {"numerical"}),
            (self.hunk("mod", [r("Evaluate", "T.ptx.st(...)", 50, op="ptx.st")], [r("Evaluate", "T.ptx.st(...)", 50, op="ptx.st")],
                       [{"slot": "value.args[0]", "path": "value.args[0].args[0].indices[1]", "old_node": "Add",
                         "new_node": "Add", "dtype": "int32", "in_index": True, "calls": ["ptx.st", "address_of"]}]),
             {"indexing"}),
        ]
        for h, want in cases:
            with self.subTest(h=h["old"][0]["text"] if h["old"] else h["new"][0]["text"]):
                self.assertEqual(set(categories(classify_ir([h]))), want)

    def test_reduction_shuffle_and_bound_are_also_numerical(self):
        r = self.rec
        shfl = self.hunk("mod", [r("Evaluate", "T.ptx.shfl_sync(...16...)", 60, reduce=True, op="ptx.shfl_sync")],
                         [r("Evaluate", "T.ptx.shfl_sync(...8...)", 60, reduce=True, op="ptx.shfl_sync")],
                         [{"slot": "value.args[2]", "path": "value.args[2].value", "old_node": "int", "new_node": "int",
                           "calls": ["ptx.shfl_sync"]}])
        self.assertEqual(set(categories(classify_ir([shfl]))), {"sync", "numerical"})
        loop = self.hunk("mod", [r("For", "for v in range(8):", 70, reduce=True)], [r("For", "for v in range(7):", 70, reduce=True)],
                         [{"slot": "end", "path": "end.value", "old_node": "int", "new_node": "int"}])
        self.assertEqual(categories(classify_ir([loop])), ["tiling", "numerical"])
        plain = self.hunk("mod", [r("For", "for v in range(8):", 70)], [r("For", "for v in range(7):", 70)],
                          [{"slot": "end", "path": "end.value", "old_node": "int", "new_node": "int"}])
        self.assertEqual(categories(classify_ir([plain])), ["tiling"])

    def test_op_vocabulary(self):
        self.assertEqual([op_class(o) for o in ("cuda.cta_sync", "ptx.bar", "ptx.shfl_sync", "ptx.cp_async", "ptx.atom",
                                                "ptx.ld", "decl_tensor", "ptr_byte_offset", "prim.exp", "ptx.fma",
                                                "cuda.block_idx")],
                         ["barrier", "barrier", "shuffle", "async", "atomic", "ldst", "alloc", "view", "math", "math",
                          "other"])

    def test_indexing_reports_as_tiling(self):
        self.assertEqual(proposal_category("indexing"), "tiling")
        self.assertEqual([proposal_category(c) for c in CATEGORIES if c != "indexing"], list(PROPOSAL_CATEGORIES))
        self.assertEqual(set(map(proposal_category, CATEGORIES)), set(PROPOSAL_CATEGORIES))
        self.assertEqual(len(PROPOSAL_CATEGORIES), 4)

    def test_tests_depend_only_on_categories(self):
        a = generate_tests("softmax", ["sync", "numerical"], [128])
        b = generate_tests("softmax", ["sync", "numerical"], [128])
        self.assertEqual([t.key() for t in a], [t.key() for t in b])


if __name__ == "__main__":
    unittest.main()
