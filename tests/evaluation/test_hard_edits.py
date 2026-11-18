"""The detection workload's generated edits (akc/evaluation/hard_edits.py): the 34 hard edits and the archive."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.hard_edits import (HARD, NO_CHANGE, in_workload, key, load_full_run,  # noqa: E402
                                       select_hard, summary)
from akc.evaluation.mutants import FRESH_LABELS, PER_OP, SETS, mutants  # noqa: E402
from akc.paths import ROOT  # noqa: E402


class HardEditTests(unittest.TestCase):
    def test_selection_from_the_full_run_gives_the_34(self):
        rows = load_full_run()
        hard = select_hard(rows)
        self.assertEqual({key(r) for r in hard}, HARD)
        self.assertEqual(len(HARD), 34)
        self.assertEqual(summary(hard), {"A0": 0, "A": 19, "B": 29, "C": 33})
        noop = [r for r in rows if not r["accept"]]
        self.assertEqual(len(noop), 27)
        self.assertEqual(summary(noop), {"A0": 0, "A": 0, "B": 0, "C": 0})

    def test_no_change_edits_are_disjoint_and_never_caught(self):
        self.assertEqual(len(NO_CHANGE), 11)
        self.assertFalse(HARD & NO_CHANGE)
        rows = {key(r): r for r in load_full_run()}
        for k in NO_CHANGE:
            self.assertFalse(any(rows[k]["detect"][c]["caught"] for c in ("A0", "A", "B", "C")), k)

    def test_generator_still_produces_every_key_and_the_workload_filter_keeps_them(self):
        kept, controls = set(), 0
        for label, _, rel in SETS["all"]:
            path = ROOT / rel
            lang = "tirx" if path.suffix == ".py" else "cuda"
            for m in mutants(path.read_text(), lang, PER_OP, extended=label in FRESH_LABELS):
                if in_workload(label, lang, m):
                    if m.accept:
                        kept.add(key(label, m))
                    else:
                        controls += 1
                        self.assertEqual(lang, "tirx")
        self.assertEqual(kept, HARD)
        self.assertEqual(controls, 27)


if __name__ == "__main__":
    unittest.main()
