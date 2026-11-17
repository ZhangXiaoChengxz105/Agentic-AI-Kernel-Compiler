"""Workloads = one operator each, described by workloads/<op>/manifest.json (data, not code).

REQUIRED manifest keys (testing works with only these + an OpSpec in task/reference.py):
  op, description, shape_keys          e.g. ["rows", "N"] or ["M", "N", "K"]
  launch                               {"kind": "row", "block": [128]} (one CTA per row) or
                                       {"kind": "tile2d", "block": [16, 16], "tile": 16} (one CTA per output tile);
                                       grid = rows (row) or ceil(N/tile) x ceil(M/tile) (tile2d). Any other
                                       launch needs OpSpec.launch.
  fixed_suite                          configuration A's shapes (power-of-two, N(0,1) data)
  variants                             [{name, lang, file (relative), role, block, bug, expect_volta}]
OPTIONAL keys (a feature without its key reports "not set up" instead of failing):
  references                           {"cuda": variant name, "tirx": variant name}: VOLTA's comparison kernel;
                                       change-aware tests compare a first candidate with it (else: all categories)
  tiles                                sizes around which boundary shapes are generated (default: the launch block / tile)
  numerical_dists                      value distributions that stress this op's numerics (default DEFAULT_DISTS)
  volta                                {"default": shape, "deep": shape | null, "max_N": int | null}  (VOLTA)
  tirx                                 {"factory_args": [...], "tensors": [...]} contract of tirx-lite modules
  perf                                 {"shape": shape} the size kernels are timed at (perf.measure, akc bench)
  signature                            kernel signature shown to the agent; its non-const pointer names VOLTA's output
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import WORKLOADS

DEFAULT_DISTS = ("large", "neg_large", "tiny", "mixed")


@dataclass(frozen=True)
class Variant:
    name: str                       # kernel entry name (CUDA extern "C") / module label (tirx)
    file: Path
    lang: str                       # "cuda" | "tirx"
    role: str                       # "reference" | "candidate"
    block: tuple[int, ...]
    bug: dict | None = None         # {"category", "change"} for injected bugs
    expect_volta: object = None     # manifest expectation (see verification.volta.layouts)


@dataclass(frozen=True)
class Workload:
    op: str
    dir: Path
    meta: dict = field(hash=False)
    variants: tuple[Variant, ...] = ()

    @property
    def shape_keys(self) -> list[str]:
        return list(self.meta["shape_keys"])

    @property
    def launch(self) -> dict:
        return self.meta["launch"]

    @property
    def tiles(self) -> list[int]:
        if self.meta.get("tiles"):
            return list(self.meta["tiles"])
        return [int(self.launch["tile"] if self.launch["kind"] == "tile2d" else self.launch["block"][0])]

    @property
    def numerical_dists(self) -> list[str]:
        return list(self.meta.get("numerical_dists") or DEFAULT_DISTS)

    @property
    def volta(self) -> dict | None:
        """The manifest's VOLTA block, or None when VOLTA is not set up for this workload."""
        return self.meta.get("volta")

    @property
    def perf_shape(self) -> dict | None:
        """The benchmark shape, or None when speed is not set up for this workload."""
        perf = self.meta.get("perf")
        return dict(perf["shape"]) if perf else None

    def has_reference(self, lang: str = "cuda") -> bool:
        return lang in (self.meta.get("references") or {})

    def launch_dims(self, shape: dict, block) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """(grid, block) of one launch: OpSpec.launch when set, else derived from the manifest's launch kind."""
        from .reference import OPS
        op = OPS.get(self.op)
        if op is not None and op.launch is not None:
            return op.launch(shape, block)
        b = tuple(int(x) for x in block) + (1,) * (3 - len(block))
        kind = self.launch["kind"]
        if kind == "row":
            return (int(shape["rows"]), 1, 1), b
        if kind == "tile2d":
            t = int(self.launch["tile"])
            return (-(-shape["N"] // t), -(-shape["M"] // t), 1), b
        raise ValueError(f"{self.op}: launch kind {kind!r} is not built in; give the OpSpec a launch function")

    def reference(self, lang: str = "cuda") -> Variant:
        name = self.meta["references"][lang]
        return self.variant(name)

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"{self.op}: no variant {name!r}")

    def candidates(self, lang: str | None = None) -> list[Variant]:
        return [v for v in self.variants if v.role == "candidate" and (lang is None or v.lang == lang)]

    def bugs(self, lang: str | None = None) -> list[Variant]:
        return [v for v in self.candidates(lang) if v.bug]

    def cap_volta_shapes(self, shapes: list[dict]) -> list[dict]:
        """Apply the manifest's VOLTA size cap (e.g. layernorm N <= 200: its VCs grow ~N^2) and dedupe."""
        mx = (self.meta.get("volta") or {}).get("max_N")
        out = []
        for s in shapes:
            s = {**s, "N": min(s["N"], mx)} if mx and "N" in s else dict(s)
            if s not in out:
                out.append(s)
        return out


def load_workload(op_or_dir: str | Path) -> Workload:
    d = Path(op_or_dir) if Path(op_or_dir).is_dir() else WORKLOADS / str(op_or_dir)
    meta = json.loads((d / "manifest.json").read_text())
    variants = []
    for v in meta["variants"]:
        block = v["block"] if isinstance(v["block"], list) else [v["block"]]
        variants.append(Variant(v["name"], d / v["file"], v.get("lang", "cuda"), v["role"],
                                tuple(int(b) for b in block), v.get("bug"), v.get("expect_volta")))
    return Workload(meta["op"], d, meta, tuple(variants))


def list_workloads() -> list[str]:
    return sorted(p.parent.name for p in WORKLOADS.glob("*/manifest.json"))
