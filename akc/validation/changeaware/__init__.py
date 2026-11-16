"""Validation mechanism 2: change-aware testing (Task 2) -- our own validator.

    diff.py       what changed between two revisions (CUDA text / tirx-lite Python AST)
    classify.py   change categories (tiling / indexing / memory / sync / numerical) with evidence
    generate.py   targeted TestCases + VOLTA shapes, driven by the workload manifest and the kernel's tiles
    validator.py  ChangeAwareValidator (common Validator interface): plan -> run tests -> findings
    hottest.py    one-shot CLI: input a kernel (+ previous version), output a problem report

plan(workload, old_src, new_src, lang=None, budget=16) -> ChangePlan (no execution; pure analysis)
"""
from __future__ import annotations

from dataclasses import dataclass

from .classify import CATEGORIES, Label, categories, classify
from .diff import Hunk, detect_lang, diff_sources
from .generate import extract_tiles, generate_tests, volta_shapes


@dataclass
class ChangePlan:
    hunks: list[Hunk]
    labels: list[Label]
    cats: list[str]
    tests: list
    volta: list[dict]
    tiles: list[int]


def plan(workload: str, old_src: str | None, new_src: str, lang: str | None = None, budget: int = 16) -> ChangePlan:
    """old_src None (no accepted version and no reference in this language) -> every category is assumed."""
    lang = lang or detect_lang(new_src)
    tiles = sorted(set(extract_tiles(new_src, lang)) | set(extract_tiles(old_src or "", lang)))
    if old_src is None:
        hunks, labels, cats = [], [], list(CATEGORIES)
    else:
        hunks = diff_sources(old_src, new_src, lang)
        labels = classify(hunks, lang)
        cats = categories(labels)
    tests = generate_tests(workload, cats, tiles, budget)
    shapes = volta_shapes(workload, cats, tiles) if lang == "cuda" else []
    return ChangePlan(hunks, labels, cats, tests, shapes, tiles)


__all__ = ["ChangePlan", "plan", "diff_sources", "classify", "categories", "generate_tests", "volta_shapes",
           "extract_tiles", "detect_lang", "CATEGORIES"]
