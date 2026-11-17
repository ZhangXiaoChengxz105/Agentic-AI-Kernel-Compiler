"""Run VOLTA (CPU only) for many candidates in parallel while the GPU validators run one at a time.

Why: in `akc mutants --detect --volta`, VOLTA is most of the time per mutant (B at the default shape, C at up to five
change-aware shapes, about 10 s per shape), and it needs no GPU. Running it serially would keep the GPU idle for hours.
The GPU validators (compiler checks, fixed tests, change-aware tests) still run serially in the main process, so their
timings are not disturbed by other GPU work.

How:
  pool = VoltaPool(workers)
  pool.submit(cand)          one task per (candidate, shape) for every shape C could ask for: the workload default plus
                             volta_shapes(workload, all categories, tiles of the candidate and its parent), the same
                             function and tiles changeaware.plan() uses, so C's shapes are a subset
  cache = pool.cache(cand)   a report cache for validation.pipeline.run_pipeline: a VOLTA lookup waits for the
                             per-shape reports of the shapes it asks for and merges them (a shape that was not
                             submitted is submitted on demand); other validators use it like a plain dict

Each worker is a separate (spawned) process with its own VOLTA build directory, so two workers never write the same
file. A merged report is what the serial validator would return for that shape list: findings in shape order (one copy
of a finding that does not depend on the shape), decided = all shapes decided, seconds = the sum of the per-shape
seconds (each includes loading the two modules, a few ms after the first shape a worker builds; the first one in a
worker also includes the tirx-lite import, about 2 s). run_pipeline lists such a report under "cached" (it came from
the cache object); its artifacts carry "parallel_pool": True.
Run it from a script file or the `akc` command: spawned workers re-import the main module, which a script read from
stdin does not have.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor

from ..validation.interface import Candidate, Finding, ValidationReport

_WORKER_VALIDATOR = None


def _worker_init(timeout_s: float) -> None:
    global _WORKER_VALIDATOR
    from ..paths import BUILD
    from ..validation.volta.validator import VoltaValidator
    _WORKER_VALIDATOR = VoltaValidator(timeout_s=timeout_s, build_root=BUILD / "volta_pool" / f"w{os.getpid()}")


def _run_shape(workload: str, lang: str, src: str, parent_src: str | None, label: str, shape: dict) -> ValidationReport:
    from ..validation.interface import ValidationContext
    parent = Candidate(workload, lang, parent_src, label=f"vp_{label}_parent") if parent_src is not None else None
    cand = Candidate(workload, lang, src, parent=parent, label=f"vp_{label}")     # own file name: the main process
    return _WORKER_VALIDATOR.validate(cand, ValidationContext(volta_shapes=[shape], device="cpu"))  # writes label_*


def _skey(shape: dict) -> str:
    return json.dumps(shape, sort_keys=True)


def superset_shapes(cand: Candidate) -> list[dict]:
    """Every shape VOLTA can be asked for in B or C for this candidate (default first), after the manifest cap."""
    from ..task.workload import load_workload
    from ..validation.changeaware.classify import CATEGORIES
    from ..validation.changeaware.generate import extract_tiles, volta_shapes
    wl = load_workload(cand.workload)
    if wl.volta is None:
        return []
    old = cand.parent.source if cand.parent is not None else ""
    tiles = sorted(set(extract_tiles(cand.source, cand.lang)) | set(extract_tiles(old or "", cand.lang)))
    return wl.cap_volta_shapes([wl.volta["default"]] + volta_shapes(cand.workload, list(CATEGORIES), tiles))


def merge(reports: list[ValidationReport], shapes: list[dict]) -> ValidationReport:
    """The report the serial VoltaValidator would give for `shapes`, built from one report per shape."""
    findings: list[Finding] = []
    seen_shapeless = set()
    for r in reports:
        for f in r.findings:
            if "shape" in (f.evidence or {}):
                findings.append(f)
            elif (f.kind, f.severity, f.message) not in seen_shapeless:   # e.g. "module failed to import": once
                seen_shapeless.add((f.kind, f.severity, f.message))
                findings.append(f)
    runs = [run for r in reports for run in r.artifacts.get("runs", [])]
    art = {"runs": runs, "shapes": shapes, "parallel_pool": True}
    if any("rewrites" in r.artifacts for r in reports):
        art["rewrites"] = next(r.artifacts["rewrites"] for r in reports if "rewrites" in r.artifacts)
    return ValidationReport("volta", not any(f.severity == "error" for f in findings),
                            all(r.decided for r in reports), findings, sum(r.seconds for r in reports), art)


class VoltaPool:
    def __init__(self, workers: int, timeout_s: float = 300.0):
        ctx = mp.get_context("spawn")       # the main process may hold a CUDA context: never fork it
        self.ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init,
                                      initargs=(timeout_s,))
        self.futures: dict[tuple[str, str], Future] = {}          # (candidate id, shape key) -> report

    @staticmethod
    def _cid(cand: Candidate) -> str:
        return f"{cand.label}|{hash((cand.source, cand.parent.source if cand.parent else None))}"

    def _submit_shape(self, cand: Candidate, shape: dict) -> Future:
        k = (self._cid(cand), _skey(shape))
        if k not in self.futures:
            self.futures[k] = self.ex.submit(_run_shape, cand.workload, cand.lang, cand.source,
                                             cand.parent.source if cand.parent else None, cand.label or "cand", shape)
        return self.futures[k]

    def submit(self, cand: Candidate) -> None:
        for s in superset_shapes(cand):
            self._submit_shape(cand, s)

    def report(self, cand: Candidate, shapes: list[dict] | None) -> ValidationReport:
        from ..task.workload import load_workload
        wl = load_workload(cand.workload)
        shapes = wl.cap_volta_shapes(shapes or [wl.volta["default"]])
        return merge([self._submit_shape(cand, s).result() for s in shapes], shapes)

    def cache(self, cand: Candidate) -> "PoolCache":
        return PoolCache(self, cand)

    def shutdown(self) -> None:
        self.ex.shutdown(wait=False, cancel_futures=True)


class PoolCache(dict):
    """run_pipeline's report cache; VOLTA lookups are answered from the pool (see the module docstring)."""

    def __init__(self, pool: VoltaPool, cand: Candidate):
        super().__init__()
        self.pool, self.cand = pool, cand

    def get(self, key, default=None):
        if key in self:
            return self[key]
        parts = json.loads(key)
        if parts.get("v") != "volta" or parts.get("src") is None:
            return default
        rep = self.pool.report(self.cand, parts.get("shapes"))
        self[key] = rep
        return rep
