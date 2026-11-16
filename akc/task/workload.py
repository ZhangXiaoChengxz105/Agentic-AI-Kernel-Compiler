"""Workloads = one operator each, described by workloads/<op>/manifest.json (data, not code).

A manifest holds everything operator-specific that code should not hard-code:
  op, description, shape_keys          e.g. ["rows", "N"] or ["M", "N", "K"]
  launch                               {"kind": "row", "block": [128]} (one CTA per row) or
                                       {"kind": "tile2d", "block": [16, 16], "tile": 16} (one CTA per output tile)
  tiles                                sizes around which boundary shapes are generated
  fixed_suite                          configuration A's shapes (power-of-two, N(0,1) data)
  numerical_dists                      value distributions that stress this op's numerics
  volta                                {"default": shape, "deep": shape | null, "max_N": int | null}
  tirx                                 {"factory_args": [...], "tensors": [...]} contract of tirx-lite modules
  perf                                 {"shape": shape} the size kernels are timed at (perf.measure, akc bench)
  references                           {"cuda": variant name, "tirx": variant name}
  variants                             [{name, lang, file (relative), role, block, bug, expect_volta}]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import WORKLOADS


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
        return list(self.meta.get("tiles", []))

    @property
    def perf_shape(self) -> dict:
        return dict(self.meta["perf"]["shape"])

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
