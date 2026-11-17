"""Task 2.2 on the IR: classify the IR hunks of one revision (irdiff.merge output) into change categories.

classify_ir(hunks, n_shapes=None) -> list[Label]     Label as in classify.py: (category, rule, line, evidence)
proposal_category(cat)                                 report view: indexing -> tiling (the proposal's 4 categories)

Input : merged IR hunks (JSON dicts: tag, old / new records with kind, text, line, feats; changes = the exact
        changed sub-nodes found with get_first_structural_mismatch, or a changed variable for an operand swap).
        Pure Python: no tvm needed here, so the worker can send hunks as JSON and tests can build them by hand.
Output: one Label per (category, rule, source line); evidence = old -> new text, the changed IR node, and the repeat
        count / shapes from the merge. Line = the old version's line for mod / del, the new version's for add.

What decides the category (the IR op that changed, not the source text):
  sync       cta_sync, bar / bar.sync, shfl_sync, mbarrier, fence, cp_async / commit / wait_group, atomics, redux:
             added, removed, moved (a delete + an add) or with changed arguments
  tiling     For.end (trip count), a loop added / removed, the device_entry grid / block, a shared-memory
             allocation's shape, cuda.smem_required
  indexing   If.condition (guard), For.begin / step, While.condition, an integer expression inside an index
             (address_of(x[...]), TensorStore.indices) or in a Bind, an integer operand swapped
  memory     ld / st form (scalar <-> vector, scope, cache modifiers, arity), allocation scope / dtype / layout,
             ptr_byte_offset / decl_tensor / reinterpret / cvta, a store into another buffer, parameter types
  numerical  exp / ex2 / max / min / add / mul / fma / rcp / sqrt calls, float arithmetic, float constants,
             Cast or dtype changes (also numerical "precision" next to the memory label of a dtype change)
Reduction rule: a changed shuffle, loop bound (For begin / end / step, While condition) or If condition whose body
(for a shuffle: its sibling statements) does a float max, min or add is ALSO numerical ("reduction-..."). A
reduction that misses or double-counts elements is wrong only for some VALUES (a partial row max overflows exp
only when the missed element is far above the rest), so it needs the value-range tests, not just boundary sizes.
In tirx-lite the Python loop of a tree reduction is unrolled at trace time, so its bound appears as If conditions
(`if thread_id < 64:`); that is why If conditions count as loop bounds here.
"""
from __future__ import annotations

import re

from .classify import Label, proposal_category  # noqa: F401  (re-exported for reports)

FLOAT_STRS = {"f16", "f32", "f64", "bf16", "f16x2", "bf16x2", "tf32", "e4m3", "e5m2"}
_LDST_STRS = re.compile(r"^(v[248]|global|shared|shared::cta|shared::cluster|local|const|param|nc|ca|cg|cs|lu|cv|wb|wt|"
                        r"L1::\w+|L2::\w+|volatile|relaxed|acquire|release|weak|b8|b16|b32|b64|b128|u8|u16|u32|u64|"
                        r"s8|s16|s32|s64)$")
_ROUND_STRS = re.compile(r"^(approx|full|rn|rz|rm|rp|rni|rzi|ftz|sat|relu)$")
SYNC_CLASSES = {"barrier": "barrier", "shuffle": "warp-shuffle", "async": "async-copy", "atomic": "atomic"}
_INT_TY = re.compile(r"^u?int\d*$|^bool$")


def op_class(op: str | None) -> str | None:
    """IR call op name (without the 'tirx.' prefix) -> barrier / shuffle / async / atomic / ldst / alloc /
    view / math / other."""
    if not op:
        return None
    o = op.lower()
    last = o.split(".")[-1]
    if "mbarrier" in o or re.match(r"(cta_sync|bar|barrier\w*|syncthreads|syncwarp|named_barrier\w*|fence\w*|"
                                   r"membar\w*|elect_sync|arrive\w*)$", last):
        return "barrier"
    if re.match(r"(shfl\w*|vote\w*|ballot\w*|redux\w*|match_\w+|activemask)$", last):
        return "shuffle"
    if "cp_async" in o or "cp.async" in o or "tma" in o or re.match(r"(commit_group|wait_group\w*|cp\w*)$", last):
        return "async"
    if re.match(r"(atom\w*|atomic\w*|red)$", last):
        return "atomic"
    if re.match(r"(ld|st|ldg|stg|lds|sts|ldu|ldmatrix|stmatrix|prefetch\w*)$", last):
        return "ldst"
    if "alloc" in last or last in ("decl_tensor", "decl_buffer"):
        return "alloc"
    if last in ("ptr_byte_offset", "reinterpret", "address_of", "cvta_generic_to_shared", "cvta", "ptr_to", "view",
                "tvm_access_ptr"):
        return "view"
    if re.match(r"(exp|exp2|exp10|ex2|log|log2|log10|lg2|sqrt|rsqrt|rcp|fma|mad|max|min|add|sub|mul|div|tanh|abs|"
                r"fabs|neg|sin|cos|pow|cvt|erf|floor|ceil|round|trunc|copysign|sigmoid|fmax|fmin)$", last):
        return "math"
    return "other"


def _reduction(rec_a: dict, rec_b: dict | None = None) -> bool:
    return bool(rec_a.get("feats", {}).get("reduce") or (rec_b or {}).get("feats", {}).get("reduce"))


def _float_text(s: str | None) -> bool:
    return bool(s) and bool(re.search(r"float|bfloat|T\.f(16|32|64)", s))


def _strip(s: str | None) -> str:
    return (s or "").strip().strip("'\"")


def record_labels(r: dict) -> list[tuple[str, str]]:
    """Categories of a whole statement that was added or removed."""
    k, f = r.get("kind"), r.get("feats", {})
    red = [("numerical", "reduction-bound")] if f.get("reduce") else []
    if k == "For":
        return [("tiling", "loop")] + red
    if k in ("If", "Else"):
        return [("indexing", "guard")] + red
    if k == "While":
        return [("indexing", "loop-bound")] + red
    if k == "Region":
        return [("tiling", "launch-geometry")]
    if k == "Function":
        return [("memory", "param-type")]
    out: list[tuple[str, str]] = []
    classes = [op_class(o) for o in ([f.get("op")] if f.get("op") else []) + list(f.get("ops", []))]
    for c in classes:
        if c in SYNC_CLASSES:
            out.append(("sync", SYNC_CLASSES[c]))
            if c == "shuffle" and f.get("reduce"):
                out.append(("numerical", "reduction-shuffle"))
    if out:
        return out
    if "alloc" in classes:
        return [("memory", "allocation")]
    if "ldst" in classes:
        return [("memory", "load/store")]
    arith = f.get("arith", [])
    if "math" in classes or any(a.endswith(":float") for a in arith) or "FloatImm" in arith or "Cast" in arith or \
            _float_text(f.get("dtype")):
        return [("numerical", "math-call" if "math" in classes else "arith")]
    if "view" in classes:
        return [("memory", "address")]
    return [("indexing", "index-arith")]


def change_labels(kind: str, ra: dict, rb: dict, ch: dict) -> list[tuple[str, str]]:
    """Categories of one exact change inside a paired (old, new) statement."""
    slot, path = ch.get("slot", ""), ch.get("path", "") or ""
    red = _reduction(ra, rb)
    if slot == "stmt":                                         # statement kind changed: both whole statements
        return record_labels(ra) + record_labels(rb)
    if kind == "Function":
        out = [("memory", "param-type")]
        return out + ([("numerical", "precision")] if "dtype" in path else [])
    if kind == "For":
        out = [{"end": ("tiling", "trip-count"), "begin": ("indexing", "loop-start"),
                "step": ("indexing", "loop-step")}.get(slot, ("tiling", "loop-kind"))]
        return out + ([("numerical", "reduction-bound")] if red else [])
    if kind in ("If", "While"):
        out = [("indexing", "guard" if kind == "If" else "loop-bound")]
        return out + ([("numerical", "reduction-bound")] if red else [])
    if kind == "Region":
        if "smem" in slot:
            return [("tiling", "smem-size")]
        return [("tiling", "launch-geometry")] if slot.startswith("args") else [("tiling", "launch-attrs")]
    if kind == "Else":
        return []
    fa, fb = ra.get("feats", {}), rb.get("feats", {})
    rec_ops = [o for o in (fa.get("op"), fb.get("op")) if o]
    path_classes = [op_class(c) for c in ch.get("calls", []) if c]
    rec_classes = [op_class(o) for o in rec_ops]
    if slot.endswith(".op"):                                  # the call itself was replaced
        olds, news = op_class(_strip(ch.get("old_text"))), op_class(_strip(ch.get("new_text")))
        for c in (olds, news):
            if c in SYNC_CLASSES:
                return [("sync", SYNC_CLASSES[c])] + ([("numerical", "reduction-shuffle")] if c == "shuffle" and red else [])
        if {olds, news} & {"ldst", "alloc", "view"}:
            return [("memory", "load/store form")]
        if {olds, news} & {"math"}:
            return [("numerical", "math-call")]
        return [("numerical", "arith")] if _float_text(ch.get("dtype")) else [("indexing", "index-arith")]
    sync = next((c for c in rec_classes + path_classes if c in SYNC_CLASSES), None)
    if sync:
        out = [("sync", SYNC_CLASSES[sync])]
        return out + ([("numerical", "reduction-shuffle")] if sync == "shuffle" and red else [])
    if slot == "operand":                                      # same text shape, another variable
        ty = ch.get("old_ty") or ch.get("new_ty") or ""
        if ch.get("in_index") or _INT_TY.match(ty):
            return [("indexing", "index-operand")]
        if "Tensor" in ty and 'scope="local"' not in ty:
            return [("memory", "buffer")]
        return [("numerical", "operand")] if _float_text(ty) else [("indexing", "operand")]
    alloc = "alloc" in rec_classes or slot == "var.ty"
    if alloc or "ty_args" in path:
        if ".shape" in path:
            shared = "shared" in (ra.get("text", "") + rb.get("text", ""))
            return [("tiling", "alloc-shape")] if shared else [("memory", "alloc-shape")]
        if ".dtype" in path:
            return [("memory", "alloc-dtype"), ("numerical", "precision")]
        if "ptr_byte_offset" in " ".join(ch.get("calls", [])):
            return [("memory", "smem-offset")]
        return [("memory", "shared/local alloc")]
    ldst = "ldst" in rec_classes + path_classes
    if ch.get("missing") and ldst:
        return [("memory", "vector-form")]
    if ch.get("old_node") == "StringImm" or ch.get("new_node") == "StringImm" or \
            ch.get("old_obj") == "StringImm" or ch.get("new_obj") == "StringImm":
        vals = {_strip(ch.get("old_text")), _strip(ch.get("new_text"))} - {""}
        out = []
        if vals & FLOAT_STRS:
            out.append(("numerical", "precision"))
            if ldst:
                out.append(("memory", "load/store form"))
        if any(_ROUND_STRS.match(v) for v in vals):
            out.append(("numerical", "rounding"))
        if any(_LDST_STRS.match(v) for v in vals) and ldst:
            out.append(("memory", "load/store form"))
        if out:
            return out
        return [("memory", "load/store form")] if ldst else [("numerical", "modifier")]
    if ch.get("in_index"):
        return [("indexing", "index-expr")]
    nodes = {ch.get("old_node"), ch.get("new_node"), ch.get("old_obj"), ch.get("new_obj")}
    if "view" in path_classes and not _float_text(ch.get("dtype")):
        return [("memory", "address")]
    if "FloatImm" in nodes:
        return [("numerical", "float-const")]
    if "Cast" in nodes:
        return [("numerical", "cast")]
    if _float_text(ch.get("dtype")) or "math" in path_classes:
        mathy = "math" in path_classes or bool(nodes & {"Call", "Max", "Min"})
        return [("numerical", "math-call" if mathy else "arith")]
    if ldst:
        return [("memory", "address")]
    return [("indexing", "index-arith")]


def _evidence(a: dict | None, b: dict | None, ch: dict | None, g: dict, n_shapes: int | None) -> str:
    from .irdiff import annotation
    """The changed IR node first (feedback shows the first 120 characters), then the statements."""
    ta = a["text"] if a else "(none)"
    tb = b["text"] if b else "(none)"
    s = f"{ta[:90]}  ->  {tb[:90]}" if a and b else (f"removed: {ta[:120]}" if a else f"added: {tb[:120]}")
    if ch:
        node = f"{ch.get('old_node')} -> {ch.get('new_node')}"
        if ch.get("old_node") == ch.get("new_node"):
            node = f"{str(ch.get('old_text', ''))[:40]} -> {str(ch.get('new_text', ''))[:40]}"
        s = f"IR {a['kind'] if a else ''}.{ch.get('path')}: {node} | {s}"
    return s + annotation(g, n_shapes)


def classify_ir(hunks: list[dict], n_shapes: int | None = None) -> list[Label]:
    labels: list[Label] = []

    def add(cat, rule, line, ev):
        labels.append(Label(cat, rule, line, ev))
    for g in hunks:
        old, new = g.get("old", []), g.get("new", [])
        paired_o = {p[0] for p in g.get("pairs", [])}
        paired_n = {p[1] for p in g.get("pairs", [])}
        by_pair: dict[tuple, list] = {}
        for ch in g.get("changes", []):
            by_pair.setdefault(tuple(ch.get("pair", (0, 0))), []).append(ch)
        for (i, j), chs in by_pair.items():
            if i >= len(old) or j >= len(new):
                continue
            ra, rb = old[i], new[j]
            line = ra.get("line") or rb.get("line")
            for ch in chs:
                for cat, rule in change_labels(ra.get("kind", ""), ra, rb, ch):
                    add(cat, rule, line, _evidence(ra, rb, ch, g, n_shapes))
        for i, r in enumerate(old):                          # deleted (or unpaired) statements
            if i not in paired_o:
                for cat, rule in record_labels(r):
                    add(cat, rule, r.get("line"), _evidence(r, None, None, g, n_shapes))
        for j, r in enumerate(new):                          # added statements
            if j not in paired_n:
                for cat, rule in record_labels(r):
                    add(cat, rule, r.get("line"), _evidence(None, r, None, g, n_shapes))
        for (i, j) in [tuple(p) for p in g.get("pairs", [])]:   # paired but no exact change found: whole records
            if (i, j) not in by_pair and i < len(old) and j < len(new) and old[i].get("text") != new[j].get("text"):
                for cat, rule in record_labels(new[j]):
                    add(cat, rule, old[i].get("line"), _evidence(old[i], new[j], None, g, n_shapes))
    seen, out = set(), []
    for lab in labels:
        k = (lab.category, lab.rule, lab.line)
        if k not in seen:
            seen.add(k)
            out.append(lab)
    return out
