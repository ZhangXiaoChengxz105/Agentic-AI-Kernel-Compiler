"""Task 2: change-aware test generation.

plan(op, old_src, new_src, block, lang=None) -> ChangePlan     (lang: "cuda" | "tirx", auto-detected)
  hunks  : what changed (diff.py, 2.1)
  labels : why it matters, with evidence (classify.py, 2.2)
  cats   : distinct categories, most evidence first
  tests  : targeted TestCases for the GPU harness (generate.py, 2.3)
  volta  : VOLTA shapes that put the boundary inside CTA 0 (generate.py, 2.3)
"""
from __future__ import annotations

from dataclasses import dataclass

from .classify import Label, categories, classify
from .diff import Hunk, detect_lang, diff_sources
from .generate import generate_tests, volta_shapes


@dataclass
class ChangePlan:
    hunks: list[Hunk]
    labels: list[Label]
    cats: list[str]
    tests: list
    volta: list[dict]


def plan(op: str, old_src: str, new_src: str, block: tuple[int, ...], budget: int = 16,
         lang: str | None = None) -> ChangePlan:
    lang = lang or detect_lang(old_src + "\n" + new_src)
    hunks = diff_sources(old_src, new_src, lang)
    labels = classify(hunks, lang)
    cats = categories(labels)
    return ChangePlan(hunks, labels, cats, generate_tests(op, cats, block, budget), volta_shapes(op, cats, block))


__all__ = ["ChangePlan", "plan", "diff_sources", "classify", "categories", "generate_tests", "volta_shapes"]
