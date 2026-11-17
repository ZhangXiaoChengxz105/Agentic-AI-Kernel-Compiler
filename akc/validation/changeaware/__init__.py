"""Validation mechanism 2: change-aware testing (Task 2) -- our own validator.

    diff.py         what changed between two revisions as source text (CUDA lines / tirx-lite Python AST)
    irdiff.py       what changed in the traced tirx-lite IR (one record per IR statement, exact changed sub-node)
    classify.py     change categories (tiling / indexing / memory / sync / numerical) of text hunks, with evidence
    classify_ir.py  the same categories from IR hunks (which IR op changed); proposal_category(): indexing -> tiling
    ir_worker.py    traces both tirx-lite versions in a child process (safety scan, 60 s timeout)
    generate.py     targeted TestCases + VOLTA shapes, driven by the workload manifest and the kernel's tiles
    validator.py    ChangeAwareValidator (common Validator interface): plan -> run tests -> findings
    hottest.py      one-shot CLI: input a kernel (+ previous version), output a problem report
    llm_classify.py optional LLM second opinion: extra categories + risky sizes (off by default; spends coins)

plan(workload, old_src, new_src, lang=None, budget=16, llm=None, classifier="auto") -> ChangePlan
  (no execution of tests; pure analysis)
  tirx-lite: the IR path. Both versions are traced at ir_worker.trace_shapes (benchmark shape with rows = 2,
             N in {1, T-1, T+1, 2T+1}, both sides of each size regime switch; <= 8 shapes), diffed (irdiff) and
             classified (classify_ir). Fallbacks, recorded in ChangePlan.analysis:
               * tracing fails / times out / is refused  -> the AST path (diff.normalize_python + classify), with the reason
               * IR equal at every shape but the AST shows a real change -> AST labels kept, ir_blind = True
               * AST says cosmetic but the IR differs -> the IR wins
  CUDA: the text path (normalized lines), unchanged.
  classifier="ast" forces the AST path for tirx (the mutation benchmark compares the two).
  old_src None, or a source that does not parse -> every category is assumed
  llm: an LlmClassifier; its labels are merged with the rules' (union) and its risky shapes become extra tests
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .classify import CATEGORIES, PROPOSAL_CATEGORIES, Label, categories, classify, proposal_category
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
    analysis: dict | None = None             # how the change was analysed (see _tirx_analysis / plan)


def plan(workload: str, old_src: str | None, new_src: str, lang: str | None = None, budget: int = 16,
         llm=None, classifier: str = "auto") -> ChangePlan:
    """old_src None (no accepted version and no reference in this language) -> every category is assumed."""
    lang = lang or detect_lang(new_src)
    tiles = sorted(set(extract_tiles(new_src, lang)) | set(extract_tiles(old_src or "", lang)))
    try:
        hunks = None if old_src is None else diff_sources(old_src, new_src, lang)
    except SyntaxError:               # tirx source that does not parse: cannot diff, so test everything
        hunks = None
    llm_hunks = hunks
    if hunks is None:
        hunks, labels, cats = [], [], list(CATEGORIES)
        analysis = {"mode": "cuda-text" if lang == "cuda" else "ast-fallback", "shapes": [],
                    "hunks": {"ast": None}, "ir_blind": False,
                    "fallback_reason": "no previous version" if old_src is None else "the source does not parse"}
    elif lang == "tirx":
        hunks, labels, analysis = _tirx_analysis(workload, old_src, new_src, hunks, classifier)
        cats = categories(labels)
        llm_hunks = llm_hunks or hunks         # the LLM sees source lines when there are any
    else:
        labels = classify(hunks, lang, (old_src or "") + "\n" + new_src)
        cats = categories(labels)
        analysis = {"mode": "cuda-text", "shapes": [], "hunks": {"text": len(hunks)}, "ir_blind": False,
                    "fallback_reason": None}
    llm_res = llm(workload, lang, old_src, new_src, llm_hunks) if (llm is not None and llm_hunks) else None
    if llm_res is not None:
        labels = labels + [lab for lab in llm_res.labels if (lab.category, lab.line) not in
                           {(x.category, x.line) for x in labels}]
        cats = categories(labels)
    from ...task.workload import load_workload
    wl = load_workload(workload)
    regimes = extract_thresholds(new_src, lang, wl.shape_keys, wl.launch["kind"], wl.domain)
    tests = generate_tests(workload, cats, tiles, budget, regimes)
    if llm_res is not None:
        tests += _llm_tests(workload, llm_res, tests)
    shapes = volta_shapes(workload, cats, tiles)          # both languages: VOLTA runs on CUDA and tirx-lite
    return ChangePlan(hunks, labels, cats, tests, shapes, tiles, llm_res.info() if llm_res is not None else None,
                      regimes, analysis)


def _tirx_analysis(workload: str, old_src: str, new_src: str, ast_hunks: list[Hunk], classifier: str):
    """-> (hunks, labels, analysis). IR path with the fallbacks listed in the module docstring."""
    ast_labels = lambda: classify(ast_hunks, "tirx", old_src + "\n" + new_src)  # noqa: E731
    if classifier == "ast":
        return ast_hunks, ast_labels(), {"mode": "ast-fallback", "shapes": [], "hunks": {"ast": len(ast_hunks)},
                                         "ir_blind": False, "fallback_reason": "AST classifier requested"}
    if old_src == new_src:                     # nothing to trace
        return [], [], {"mode": "ir", "shapes": [], "hunks": {"ir": 0, "ast": 0}, "ir_blind": False,
                        "fallback_reason": None, "identical_sources": True}
    from . import ir_worker
    from .irdiff import to_hunks
    t0 = time.perf_counter()
    shapes = ir_worker.trace_shapes(workload, old_src, new_src)
    res = ir_worker.analyze(workload, old_src, new_src, shapes)
    analysis = {"mode": "ir", "shapes": res.get("shapes", shapes), "ir_blind": False, "fallback_reason": None,
                "hunks": {"ir": len(res.get("hunks") or []), "ir_per_shape": res.get("per_shape", {}),
                          "ast": len(ast_hunks)},
                "skipped_shapes": res.get("skipped", {}), "seconds": round(time.perf_counter() - t0, 2)}
    if not res.get("ok"):
        analysis.update(mode="ast-fallback", shapes=[], fallback_reason=res.get("error") or "IR analysis failed")
        return ast_hunks, ast_labels(), analysis
    ir_hunks = res.get("hunks") or []
    if not ir_hunks:
        if ast_hunks:                          # the source changed but no traced shape shows it in the IR
            analysis.update(ir_blind=True, fallback_reason="IR identical at every traced shape; AST labels kept")
            return ast_hunks, ast_labels(), analysis
        return [], [], analysis
    labels = [Label(**lab) for lab in res.get("labels", [])]
    if not ast_hunks:
        analysis["ast_cosmetic"] = True        # the AST called it cosmetic; the IR differs and wins
    return to_hunks(ir_hunks, len(analysis["shapes"])), labels, analysis


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
           "extract_tiles", "extract_thresholds", "detect_lang", "CATEGORIES", "PROPOSAL_CATEGORIES",
           "proposal_category"]
