"""Validation mechanism 2: change-aware testing (Task 2) -- our own validator.

    diff.py       what changed between two revisions (CUDA text / tirx-lite Python AST)
    classify.py   change categories (tiling / indexing / memory / sync / numerical) with evidence
    generate.py   targeted TestCases + VOLTA shapes, driven by the workload manifest and the kernel's tiles
    validator.py  ChangeAwareValidator (common Validator interface): plan -> run tests -> findings
    hottest.py    one-shot CLI: input a kernel (+ previous version), output a problem report
    llm_classify.py  optional LLM second opinion: extra categories + risky sizes (off by default; spends coins)

plan(workload, old_src, new_src, lang=None, budget=16, llm=None) -> ChangePlan (no execution; pure analysis)
  old_src None, or a source that does not parse -> every category is assumed
  llm: an LlmClassifier; its labels are merged with the rules' (union) and its risky shapes become extra tests
"""
from __future__ import annotations

from dataclasses import dataclass

from .classify import CATEGORIES, Label, categories, classify
from .diff import Hunk, detect_lang, diff_sources
from .generate import extract_thresholds, extract_tiles, generate_tests, volta_shapes


@dataclass
class ChangePlan:
    hunks: list[Hunk]
    labels: list[Label]
    cats: list[str]
    tests: list
    volta: list[dict]
    tiles: list[int]
    llm: dict | None = None                  # LlmResult.info() when an LLM was asked
    regimes: list | None = None              # [(shape key, v)]: the kernel switches code path between v and v+1


def plan(workload: str, old_src: str | None, new_src: str, lang: str | None = None, budget: int = 16,
         llm=None) -> ChangePlan:
    """old_src None (no accepted version and no reference in this language) -> every category is assumed."""
    lang = lang or detect_lang(new_src)
    tiles = sorted(set(extract_tiles(new_src, lang)) | set(extract_tiles(old_src or "", lang)))
    try:
        hunks = None if old_src is None else diff_sources(old_src, new_src, lang)
    except SyntaxError:               # tirx source that does not parse: cannot diff, so test everything
        hunks = None
    if hunks is None:
        hunks, labels, cats = [], [], list(CATEGORIES)
    else:
        labels = classify(hunks, lang, (old_src or "") + "\n" + new_src)
        cats = categories(labels)
    llm_res = llm(workload, lang, old_src, new_src, hunks) if (llm is not None and hunks) else None
    if llm_res is not None:
        labels = labels + [lab for lab in llm_res.labels if (lab.category, lab.line) not in
                           {(x.category, x.line) for x in labels}]
        cats = categories(labels)
    from ...task.workload import load_workload
    wl = load_workload(workload)
    regimes = extract_thresholds(new_src, lang, wl.shape_keys, wl.launch["kind"])
    tests = generate_tests(workload, cats, tiles, budget, regimes)
    if llm_res is not None:
        tests += _llm_tests(workload, llm_res, tests)
    shapes = volta_shapes(workload, cats, tiles) if lang == "cuda" else []
    return ChangePlan(hunks, labels, cats, tests, shapes, tiles, llm_res.info() if llm_res is not None else None,
                      regimes)


def _llm_tests(workload: str, res, existing: list) -> list:
    from ...task.testcase import TestCase
    from .generate import _seed
    out = []
    for shape, cat, why in res.shapes:
        if any(t.shape == shape and t.dist == "normal" for t in existing + out):
            continue
        out.append(TestCase(workload, shape, "normal", _seed(workload, "llm", sorted(shape.items())), 1,
                            f"{cat}: LLM-suggested size {shape}: {why}"))
    return out


__all__ = ["ChangePlan", "plan", "diff_sources", "classify", "categories", "generate_tests", "volta_shapes",
           "extract_tiles", "extract_thresholds", "detect_lang", "CATEGORIES"]
