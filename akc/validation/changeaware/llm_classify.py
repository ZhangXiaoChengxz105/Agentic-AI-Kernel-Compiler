"""Optional LLM second opinion for change-aware testing (OFF by default; every uncached call spends Lumen coins).

The rule classifier (classify.py) is fast, free and deterministic. What it cannot do is reason about WHEN an edit
matters: `while (N - i > 384)` -> `> 383` is "indexing", but only sizes N = 384 + tid + 512k expose it, and the
generic boundary sizes may miss them. The LLM is asked for two things:
  1. categories per changed line (merged with the rules' labels as a UNION: it can only add tests, never remove);
  2. up to 4 concrete "risky shapes" where this edit is most likely to misbehave (validated, then run as tests).

LlmClassifier(model="glm-5.3", reasoning_effort="low")(workload, lang, old_src, new_src, hunks) -> LlmResult
  * prompt = category definitions + the normalized diff + the new kernel with line numbers (no old kernel)
  * reply must be one JSON object; anything unusable -> LlmResult.error, no labels (the rules still apply)
  * cached on disk by a hash of (model, effort, messages): re-running an experiment costs nothing
  * every call is logged by the Lumen client (logs/lumen/calls.jsonl); cost and latency are reported in the
    change-aware report's artifacts["llm"] (part of configuration C's validation overhead)
Use: ValidationContext(extra={"llm_classifier": LlmClassifier()}), or `--llm-classify` on akc validate / hottest /
mutants. Not used in configurations A or B.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

from ...paths import BUILD
from ...task.workload import load_workload
from .classify import CATEGORIES, Label
from .diff import Hunk

MAX_SHAPES = 4
DIM_CAP = {"rows": 256, "N": 16384, "M": 2048, "K": 4096}

SYSTEM = """You review one revision of a GPU kernel inside a kernel compiler. An agent edited the kernel; you see \
the diff between the last accepted version and the new revision. Do NOT judge whether the new code is correct. \
Say what KIND of change each edit is, so that targeted tests can be generated, and name input sizes at which the \
edit is most likely to misbehave.

Categories (one edit may have several):
- tiling: tile / block sizes, launch geometry, number of tiles or loop trip counts, rounding sizes to tile multiples, unrolling.
- indexing: loop bounds, starts and steps; guards (if conditions); comparison operators; +1/-1; array index expressions; index or pointer arithmetic.
- memory: shared / local memory declarations and sizes, vector loads / stores, alignment, load / store forms.
- sync: barriers, warp shuffles and votes, atomics, async copies: anything about which thread sees which value when.
- numerical: math functions, float arithmetic and accumulation order, initial values, special constants (inf, eps), precision.

Reply with ONE JSON object and nothing else:
{"changes": [{"line": <line number in the NEW version>, "categories": ["..."], "why": "<= 15 words"}],
 "risky_shapes": [{"shape": {...}, "category": "...", "why": "<= 15 words"}]}
Give at most 4 risky_shapes, only sizes where THIS edit changes behaviour (e.g. where a changed bound or branch flips)."""


@dataclass
class LlmResult:
    labels: list[Label] = field(default_factory=list)
    shapes: list[tuple[dict, str, str]] = field(default_factory=list)   # (shape, category, why)
    model: str = ""
    cost: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    error: str = ""

    def info(self) -> dict:
        return {"model": self.model, "cost": self.cost, "latency_s": self.latency_s, "cached": self.cached,
                "error": self.error, "n_labels": len(self.labels), "shapes": [s for s, _, _ in self.shapes]}


def _diff_text(hunks: list[Hunk]) -> str:
    out = []
    for h in hunks:
        out += [f"- old L{n}: {t}" for n, t in h.old] + [f"+ new L{n}: {t}" for n, t in h.new]
        out.append("")
    return "\n".join(out).strip()


def build_messages(workload: str, lang: str, new_src: str, hunks: list[Hunk]) -> list[dict]:
    wl = load_workload(workload)
    keys = ", ".join(f'"{k}" (1..{DIM_CAP.get(k, 4096)})' for k in wl.shape_keys)
    numbered = "\n".join(f"{i:4d}  {line}" for i, line in enumerate(new_src.splitlines(), 1))
    user = (f"Operator: {workload}: {wl.meta.get('description', '')}\nLanguage: {lang}. Launch: {json.dumps(wl.launch)}. "
            f"Shape keys for risky_shapes: {keys}.\n\nDiff (normalized: comments and spacing removed):\n"
            f"{_diff_text(hunks)}\n\nNew version with line numbers:\n{numbered}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_reply(text: str, workload: str) -> tuple[list[Label], list[tuple[dict, str, str]]]:
    """Validate the model's JSON; raise ValueError when nothing usable is in it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    data = json.loads(text[start:end + 1])
    wl = load_workload(workload)
    labels = []
    for ch in data.get("changes") or []:
        cats = [c for c in (ch.get("categories") or []) if c in CATEGORIES]
        line = ch.get("line") if isinstance(ch.get("line"), int) else None
        why = str(ch.get("why", ""))[:200]
        labels += [Label(c, "llm", line, why) for c in cats]
    shapes = []
    for rs in (data.get("risky_shapes") or [])[:MAX_SHAPES]:
        raw = rs.get("shape") if isinstance(rs, dict) else None
        if not isinstance(raw, dict):
            continue
        shape = {}
        for k in wl.shape_keys:
            v = raw.get(k, 2 if k == "rows" else None)
            if not isinstance(v, int) or not 1 <= v <= DIM_CAP.get(k, 4096):
                shape = None
                break
            shape[k] = v
        if shape:
            cat = rs.get("category") if rs.get("category") in CATEGORIES else "llm"
            shapes.append((shape, cat, str(rs.get("why", ""))[:200]))
    if not labels and not shapes and (data.get("changes") or data.get("risky_shapes")):
        raise ValueError("reply had entries but none were valid")
    return labels, shapes


class LlmClassifier:
    def __init__(self, model: str = "glm-5.3", reasoning_effort: str = "low", max_tokens: int = 4096,
                 client=None, cache_dir=None, use_cache: bool = True):
        self.model, self.effort, self.max_tokens = model, reasoning_effort, max_tokens
        self._client, self.use_cache = client, use_cache
        self.cache_dir = cache_dir or (BUILD / "llm_cache" / "changeaware")

    @property
    def client(self):
        if self._client is None:
            from ...llm.lumen import LumenClient
            self._client = LumenClient(model=self.model, timeout_s=300)
        return self._client

    def __call__(self, workload: str, lang: str, old_src: str, new_src: str, hunks: list[Hunk]) -> LlmResult:
        res = LlmResult(model=self.model)
        if not hunks:
            return res                                        # nothing changed (or only cosmetics): no call
        messages = build_messages(workload, lang, new_src, hunks)
        extra = {} if self.effort in ("", "none") else {"reasoning_effort": self.effort}
        key = hashlib.sha256(json.dumps([self.model, self.effort, self.max_tokens, messages]).encode()).hexdigest()[:32]
        path = self.cache_dir / f"{key}.json"
        t0 = time.perf_counter()
        try:
            if self.use_cache and path.exists():
                rec = json.loads(path.read_text())
                res.cached = True
            else:
                r = self.client.chat(messages, max_tokens=self.max_tokens, temperature=0.0,
                                     tag=f"changeaware-{workload}", **extra)
                rec = {"content": r.content, "cost": r.cost, "latency_s": r.latency_s, "model": r.model,
                       "finish_reason": r.finish_reason, "dry_run": r.dry_run}
                res.cost = r.cost
                if self.use_cache and not r.dry_run:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(rec))
            res.labels, res.shapes = parse_reply(rec["content"], workload)
        except Exception as e:  # noqa: BLE001  (network, budget, consent, bad JSON: the rules still apply)
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
        res.latency_s = time.perf_counter() - t0
        return res
