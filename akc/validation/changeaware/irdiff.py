"""Task 2.1 on the IR: what changed between two tirx-lite revisions, read from the traced TIRx IR.

The proposal asks to "inspect the modified IR operations". tirx-lite plays CAKE's role, so each revision is traced
(make_kernel(rows, n) -> txl.Kernel.func, a TIRx Function) and the two Functions are compared, at several shapes
(ir_worker.py traces them in a child process; changeaware.plan picks the shapes). Needs tvm_tirx; every tvm import
is local, so this module can be imported without it.

flatten(func, src_file) -> Flat(recs, names)
    One Rec per statement or compound-statement header, in program order:
      depth, kind (Bind / Evaluate / TensorStore / For / If / Else / While / Region / Function), canonical text,
      line, caller line, the IR node, and features for the classifier (call ops, float reduction nearby, ...).
    Canonical text = the TVMScript printer's line for that statement (one print of the whole function, each
    statement tagged with obj_to_annotate), with
      * variables renamed in definition order: txl names every local `v`, so they become v0, v1, ...; parameters
        and txl's named values (bx, thread_id, ...) keep their names; the printer's de-duplication suffix
        (`v3_1`) is dropped;
      * the free-variable preamble (`v3_1 = T.Var(...)` lines before `def`) dropped;
      * if printing fails: the statement's own script(), then str(stmt), then "<type> #<structural_hash>".
    line / caller: the statement's source location in the kernel module (SourceLoc.start_line). A statement made
    inside a helper function carries CallSiteLoc(callee, caller): line = the helper's line, caller = the
    outermost call's line in the module. Lines in other files (the txl library) are skipped.
diff_records(a, b) -> list[hunk dict]
    difflib over (depth, kind, text with variable numbers removed). Aligning WITHOUT variable numbers keeps one
    inserted local from renumbering every later statement. Variable identity is then checked separately (a
    definition-order bijection built from aligned pairs): an aligned pair whose variables do not correspond
    (max(a, b) -> max(b, a), a store into another buffer) becomes a "mod" hunk too.
    For every replaced pair, tvm_ffi.get_first_structural_mismatch(map_free_vars=True) on each field / call
    argument ("slot") gives the exact changed sub-node, e.g. If.condition LT -> LE, For.end 2 -> 1,
    value.b.args[0] Sub -> TensorLoad.
merge(per_shape) -> list[merged hunk dict]
    Hunks of all shapes grouped by (tag, source lines, kinds): an unrolled loop repeats one source edit (nosync:
    14 deleted barriers) and the same edit shows at several shapes; each group becomes one hunk with a count
    ("x14") and the shapes where it appears (a dropped tail only shows at N that are not tile multiples).
to_hunks(merged, n_shapes) -> list[diff.Hunk]      the existing Hunk objects, line numbers from the source locs
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .diff import Hunk

COMPOUND = ("For", "If", "While", "RegionStmt")
FLOAT_STRS = {"f16", "f32", "f64", "bf16", "f16x2", "bf16x2", "tf32"}
_IDENT = re.compile(r"[A-Za-z_]\w*")
_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*(?:[eE][-+]?\d+)?|\S")
_ANN_LEAF = re.compile(r"^(?P<code>.*?)\s+# @(?P<id>\d+)$")
_ANN_HEAD = re.compile(r"^\s*# @(?P<id>\d+)$")
_PREAMBLE = re.compile(r"^[A-Za-z_]\w* = T\.(?:Var|dynamic)\(")
_ARITH = {"Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod", "Min", "Max", "LShift", "RShift",
          "BitwiseAnd", "BitwiseOr", "BitwiseXor"}
_REDUCE_NODES = {"Add", "Max", "Min"}
_REDUCE_CALLS = re.compile(r"^(?:ptx\.)?(?:max|min|add|redux\w*)$")


@dataclass
class Rec:
    depth: int
    kind: str
    text: str
    key: str = ""
    line: int | None = None
    caller: int | None = None
    defs: tuple = ()
    feats: dict = field(default_factory=dict)
    node: object = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {"depth": self.depth, "kind": self.kind, "text": self.text, "line": self.line,
                "caller": self.caller, "feats": self.feats}


@dataclass
class Flat:
    recs: list
    names: dict                       # canonical name -> {"hint": original name, "ty": type text}
    params: tuple = ()
    printer: str = "script"           # "script" | "fallback" (some statement printed with a fallback)


# ---- helpers on IR objects ----------------------------------------------------------------------------------
def _tname(x) -> str:
    return type(x).__name__


def _op_name(call) -> str:
    op = getattr(call, "op", None)
    name = getattr(op, "name", None) or str(op)
    return name[5:] if name.startswith("tirx.") else name


def _dtype(x) -> str | None:
    ty = getattr(x, "ty", None)
    if ty is not None and _tname(ty) == "PrimType":
        return str(ty)
    return None


def _is_float(dt: str | None) -> bool:
    return bool(dt) and dt.startswith(("float", "bfloat"))


def _walk(obj, cb) -> None:
    import tvm_ffi
    try:
        tvm_ffi.structural_walk(obj, cb)
    except Exception:  # noqa: BLE001  (a value the walker cannot descend into)
        pass


def _call_strs(call) -> list[str]:
    return [a.value for a in getattr(call, "args", []) if _tname(a) == "StringImm"]


def features(objs) -> dict:
    """Call ops, string immediates of those calls (ld/st forms, dtypes, shuffle modes), arithmetic node kinds."""
    ops, strs, arith = set(), set(), set()

    def cb(v):
        n = _tname(v)
        if n == "Call":
            ops.add(_op_name(v))
            strs.update(s for s in _call_strs(v) if s)
        elif n in _ARITH:
            arith.add(f"{n}:{'float' if _is_float(_dtype(v)) else 'int'}")
        elif n == "Cast":
            arith.add("Cast")
        elif n == "FloatImm":
            arith.add("FloatImm")
    for o in objs:
        if o is not None and not isinstance(o, (int, float, str)):
            _walk(o, cb)
    return {"ops": sorted(ops), "strs": sorted(strs), "arith": sorted(arith)}


def float_reduce(obj) -> bool:
    """Does obj contain a float max / min / add (a reduction step), as an expression or a ptx call?"""
    found = []

    def cb(v):
        if found:
            return None
        n = _tname(v)
        if n in _REDUCE_NODES and _is_float(_dtype(v)):
            found.append(n)
        elif n == "Call" and _REDUCE_CALLS.match(_op_name(v)) and (
                set(_call_strs(v)) & FLOAT_STRS or _is_float(_dtype(v))):
            found.append(_op_name(v))
        return None
    _walk(obj, cb)
    return bool(found)


def _loc_lines(loc, src_file: str | None) -> tuple[int | None, int | None]:
    """(line, caller) in the module file: innermost frame there, and the outermost call site if different."""
    frames, cur = [], loc
    while cur is not None:
        n = _tname(cur)
        if n == "CallSiteLoc":
            frames.append(cur.callee)
            cur = cur.caller
        else:
            frames.append(cur)
            cur = None
    lines = []
    for f in frames:
        while _tname(f) == "CallSiteLoc":            # a callee that is itself a call site: take its innermost frame
            f = f.callee
        if _tname(f) != "SourceLoc":
            continue
        name = getattr(getattr(f, "source_name", None), "name", None)
        if src_file is None or name == src_file:
            lines.append(int(f.start_line))
    if not lines:
        return None, None
    return lines[0], (lines[-1] if len(lines) > 1 and lines[-1] != lines[0] else None)


# ---- canonical naming -------------------------------------------------------------------------------------
def _canonical(func):
    """Rename every non-parameter variable in definition order; returns (renamed func, names table)."""
    import tvm_ffi
    from tvm_tirx.ir import Var
    order, seen = [], set()

    def visit(v):
        h = v.__chandle__()
        if h not in seen:
            seen.add(h)
            order.append(v)
    tvm_ffi.structural_walk(func, (Var, visit), order="pre")
    params = {p.__chandle__(): p.name for p in func.params}
    names: dict[str, dict] = {}
    new_name: dict[int, str] = {}
    counters: dict[str, int] = {}
    for v in order:
        h = v.__chandle__()
        if h in params:
            names[params[h]] = {"hint": v.name, "ty": str(v.ty)}
            continue
        hint = v.name or "v"
        if hint != "v" and hint not in names:
            name = hint
        else:
            while True:
                k = counters.get(hint, 0)
                counters[hint] = k + 1
                name = f"{hint}{k}"
                if name not in names and name not in params.values():
                    break
        new_name[h] = name
        names[name] = {"hint": hint, "ty": str(v.ty)}
    memo: dict[int, object] = {}

    def ren(v):
        h = v.__chandle__()
        if h not in new_name:
            return v
        if h not in memo:
            memo[h] = Var(new_name[h], v.ty, v.loc)
        return memo[h]
    return tvm_ffi.structural_map(func, (Var, ren)), names


def _clean(text: str, names: dict) -> str:
    """Drop printer de-duplication suffixes (v3_1 -> v3) and collapse whitespace."""
    def fix(m):
        w = m.group(0)
        if w in names:
            return w
        base, _, num = w.rpartition("_")
        return base if num.isdigit() and base in names else w
    return " ".join(_IDENT.sub(fix, text).split())


def _key(text: str, names: dict) -> str:
    return _IDENT.sub(lambda m: names[m.group(0)]["hint"] if m.group(0) in names else m.group(0), text)


def _short(obj, names: dict | None = None, limit: int = 160) -> str:
    """One-line text of an IR sub-node (printer, then str, then type + hash)."""
    if obj is None or isinstance(obj, (int, float, str)):
        return repr(obj)
    for fn in (lambda o: o.script(), str):
        try:
            lines = [ln for ln in fn(obj).splitlines() if ln.strip() and not ln.startswith(("#", "from __future__"))]
            while len(lines) > 1 and _PREAMBLE.match(lines[0]):
                lines.pop(0)
            t = " ; ".join(ln.strip() for ln in lines)
            return (_clean(t, names) if names else t)[:limit]
        except Exception:  # noqa: BLE001
            continue
    return _hash_text(obj)


def _hash_text(obj) -> str:
    import tvm_ffi
    try:
        return f"{_tname(obj)} #{tvm_ffi.structural_hash(obj)}"
    except Exception:  # noqa: BLE001
        return f"{_tname(obj)} #?"


# ---- flatten ----------------------------------------------------------------------------------------------
def _kind(node) -> str:
    n = _tname(node)
    return "Region" if n == "RegionStmt" else n


def _header_parts(node) -> list:
    n = _tname(node)
    if n == "For":
        return [node.begin, node.end, node.step]
    if n in ("If", "While"):
        return [node.condition]
    if n == "RegionStmt":
        return list(node.args)
    return [node]


def _collect(stmt, depth: int, out: list, sib_reduce: bool = False) -> None:
    """Program-order items: (kind, node, depth, reduce flag). Else markers have node = the If."""
    n = _tname(stmt)
    if n == "SeqStmt":
        kids = list(stmt.seq)
        leaf_red = any(float_reduce(k) for k in kids if _tname(k) not in COMPOUND + ("SeqStmt",))
        for k in kids:
            _collect(k, depth, out, leaf_red)
        return
    if n == "For" or n == "While":
        out.append((_kind(stmt), stmt, depth, float_reduce(stmt.body)))
        _collect(stmt.body, depth + 1, out)
    elif n == "If":
        out.append(("If", stmt, depth, float_reduce(stmt.then_case)))
        _collect(stmt.then_case, depth + 1, out)
        if stmt.else_case is not None:
            out.append(("Else", stmt, depth, float_reduce(stmt.else_case)))
            _collect(stmt.else_case, depth + 1, out)
    elif n == "RegionStmt":
        out.append(("Region", stmt, depth, False))
        _collect(stmt.body, depth + 1, out)
    else:
        out.append((_kind(stmt), stmt, depth, sib_reduce or float_reduce(stmt)))


def _defs(node, kind: str) -> tuple:
    if kind == "Bind":
        return (node.var.name,)
    if kind == "For":
        return (node.var.name,)
    if kind == "Region":
        return tuple(v.name for v in list(node.body_params) + list(node.result_vars))
    return ()


def _feats(kind: str, node, reduce: bool) -> dict:
    parts = [] if kind == "Else" else _header_parts(node) if kind in ("For", "If", "While", "Region") else [node]
    f = features(parts)
    f["reduce"] = bool(reduce)
    if kind in ("Evaluate", "Bind"):
        v = node.value
        if _tname(v) == "Call":
            f["op"] = _op_name(v)
    if kind == "TensorStore":
        f["dtype"] = _dtype(node.value)
    elif kind == "Bind":
        f["dtype"] = _dtype(node.value) or str(node.var.ty)
    return f


def _stmt_text(kind: str, node, names: dict) -> str:
    """Fallback printing of one statement: its own script() (first line for a header), str(), type + hash."""
    if kind == "Else":
        return "else:"
    for fn in (lambda o: o.script(), str):
        try:
            lines = [ln for ln in fn(node).splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            while len(lines) > 1 and _PREAMBLE.match(lines[0]):
                lines.pop(0)
            if not lines:
                continue
            t = lines[0] if kind in ("For", "If", "While", "Region") else " ; ".join(ln.strip() for ln in lines)
            return _clean(t, names)
        except Exception:  # noqa: BLE001
            continue
    return _hash_text(node)


def flatten(func, src_file: str | None = None) -> Flat:
    """One record per statement / header of a traced tirx-lite Function (see the module docstring)."""
    g, names = _canonical(func)
    items: list = []
    _collect(g.body, 0, items)
    ann = {}
    for i, (kind, node, _, _) in enumerate(items):
        if kind != "Else" and node not in ann:
            ann[node] = f"@{i}"
    texts: dict[int, str] = {}
    printer = "script"
    try:
        script = g.script(obj_to_annotate=ann)
        texts = _parse(script, items, names)
    except Exception:  # noqa: BLE001  (printer failure: every statement falls back to its own text)
        texts = {}
    recs = []
    sig = ", ".join(f"{p.name}: {names.get(p.name, {}).get('ty', '?')}" for p in g.params)
    recs.append(Rec(0, "Function", f"def KERNEL({sig}):", f"def KERNEL({sig}):", None, None, (), {"ops": []}, g))
    for i, (kind, node, depth, red) in enumerate(items):
        text = texts.get(i)
        if text is None:
            printer = "fallback"
            text = _stmt_text(kind, node, names)
        line, caller = _loc_lines(getattr(node, "loc", None), src_file) if kind != "Else" else (None, None)
        recs.append(Rec(depth + 1, kind, text, _key(text, names), line, caller,
                        _defs(node, kind), _feats(kind, node, red), node))
    for i, r in enumerate(recs):              # an else marker takes the line of the first statement it holds
        if r.kind == "Else" and i + 1 < len(recs):
            r.line, r.caller = recs[i + 1].line, recs[i + 1].caller
    return Flat(recs, names, tuple(p.name for p in g.params), printer)


def _parse(script: str, items: list, names: dict) -> dict[int, str]:
    """Map annotated printer lines back to items: '... # @i' is statement i; '# @i' + next line is header i;
    'else:' lines go to the next Else marker in program order."""
    lines = script.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("def ")) + 1
    except StopIteration:
        return {}
    else_queue = [i for i, it in enumerate(items) if it[0] == "Else"]
    ids_by_node: dict = {}
    for i, (kind, node, _, _) in enumerate(items):
        if kind != "Else":
            ids_by_node.setdefault(node, []).append(i)
    texts: dict[int, str] = {}
    used: set[int] = set()

    def target(aid: int) -> int:
        """The annotation id is the node's first item; a node used twice gets its later items in order."""
        node = items[aid][1]
        for i in ids_by_node.get(node, [aid]):
            if i not in used:
                used.add(i)
                return i
        return aid
    pending = None
    last = None
    for ln in lines[start:]:
        if not ln.strip():
            continue
        m = _ANN_HEAD.match(ln)
        if m:
            pending = target(int(m.group("id")))
            continue
        if pending is not None:
            texts[pending] = _clean(ln.strip(), names)
            last, pending = None, None
            continue
        m = _ANN_LEAF.match(ln)
        if m:
            i = target(int(m.group("id")))
            texts[i] = _clean(m.group("code").strip(), names)
            last = i
            continue
        if ln.strip() == "else:" and else_queue:
            texts[else_queue.pop(0)] = "else:"
            last = None
            continue
        if last is not None:                         # continuation of a multi-line statement
            texts[last] = _clean(texts[last] + " " + ln.strip(), names)
    return texts


# ---- diff -------------------------------------------------------------------------------------------------
def _align_key(r: Rec) -> str:
    return f"{r.depth}|{r.kind}|{r.key}"


def _bind_defs(a: Rec, b: Rec, fwd: dict, back: dict) -> None:
    for x, y in zip(a.defs, b.defs):
        if x not in fwd and y not in back:
            fwd[x], back[y] = y, x


def _operand_diff(a: Rec, b: Rec, na: dict, nb: dict, fwd: dict, back: dict):
    """None if the texts agree under the variable bijection (binding unmapped variables on first use, like
    map_free_vars); else (old token, new token) of the first disagreement."""
    ta, tb = _TOKEN.findall(a.text), _TOKEN.findall(b.text)
    if len(ta) != len(tb):
        return ("<length>", "<length>")
    new_f, new_b = {}, {}
    for x, y in zip(ta, tb):
        if x in na and y in nb:
            fx = fwd.get(x, new_f.get(x))
            by = back.get(y, new_b.get(y))
            if fx is None and by is None:
                new_f[x], new_b[y] = y, x
                continue
            if fx != y or by != x:
                return (x, y)
        elif x != y:
            return (x, y)
    fwd.update(new_f)
    back.update(new_b)
    return None


def _slots(r: Rec) -> dict:
    n, k = r.node, r.kind
    try:
        if k == "Function":
            return {f"params[{i}].ty": p.ty for i, p in enumerate(n.params)}
        if k == "For":
            return {"begin": n.begin, "end": n.end, "step": n.step, "kind": int(n.kind)}
        if k in ("If", "While"):
            return {"condition": n.condition}
        if k == "Region":
            out = {"op": _op_name(n)} | {f"args[{i}]": a for i, a in enumerate(n.args)}
            try:
                out |= {f"attrs[{key}]": val for key, val in dict(n.attrs).items()}
            except Exception:  # noqa: BLE001
                out["attrs"] = n.attrs
            return out
        if k == "Bind":
            return {"var.ty": n.var.ty} | _expr_slots("value", n.value)
        if k == "TensorStore":
            return {"dest": n.dest} | {f"indices[{i}]": e for i, e in enumerate(n.indices)} | \
                _expr_slots("value", n.value)
        if k == "Evaluate":
            return _expr_slots("value", n.value)
    except Exception:  # noqa: BLE001
        pass
    return {"stmt": n}


def _expr_slots(prefix: str, e) -> dict:
    if _tname(e) == "Call":
        out = {f"{prefix}.op": _op_name(e)} | {f"{prefix}.args[{i}]": a for i, a in enumerate(e.args)}
        out |= {f"{prefix}.ty_args[{i}]": t for i, t in enumerate(getattr(e, "ty_args", []) or [])}
        return out
    return {prefix: e}


def _is_obj(x) -> bool:
    import tvm_ffi
    return isinstance(x, tvm_ffi.Object)


def _step(node, s):
    kind, key = int(s.kind), s.key
    if kind == 0:
        return getattr(node, key), f".{key}"
    if kind == 1:
        return node[key], f"[{key}]"
    if kind == 2:
        return node[key], f"[{key!r}]"
    raise LookupError(kind)


def _navigate(root, path) -> tuple[list, str, bool]:
    """Follow an AccessPath from root: ([(step text, node)], path text, missing item?)."""
    chain, text, missing, node = [], "", False, root
    for s in path.to_steps():
        if int(s.kind) >= 3:                     # *_MISSING: one side has fewer items / no such attribute
            missing = True
            text += f"[{s.key}]" if int(s.kind) == 4 else f".{s.key}"
            break
        try:
            node, st = _step(node, s)
        except Exception:  # noqa: BLE001
            break
        text += st
        chain.append((st, node))
    return chain, text, missing


def _describe(node, names: dict) -> tuple[str, str]:
    if _is_obj(node):
        return _tname(node), _short(node, names)
    return type(node).__name__, repr(node)


def _detail(slot: str, x, y, mm, na: dict, nb: dict, rec_op: str | None) -> dict:
    ca, path, miss_a = _navigate(x, mm[0])
    cb, _, miss_b = _navigate(y, mm[1])
    last_a = ca[-1][1] if ca else x
    last_b = cb[-1][1] if cb else y
    objs_a = [x] + [n for _, n in ca if _is_obj(n)]
    objs_b = [y] + [n for _, n in cb if _is_obj(n)]
    oa = objs_a[-1]
    ob = objs_b[-1]
    calls = ([rec_op] if rec_op and slot.startswith("value.args") else []) + \
        [_op_name(n) for n in objs_a + objs_b if _tname(n) == "Call"]
    dt = next((d for d in (_dtype(n) for n in reversed(objs_a + objs_b)) if d), None)
    old_node, old_text = _describe(last_a, na)
    new_node, new_text = _describe(last_b, nb)
    steps = slot + path
    return {"slot": slot, "path": steps, "ctx": [_tname(n) for n in objs_a],
            "old_node": old_node if not miss_a else "missing", "new_node": new_node if not miss_b else "missing",
            "old_obj": _tname(oa), "new_obj": _tname(ob),
            "old_text": old_text, "new_text": new_text, "dtype": dt,
            "in_index": ".indices" in steps or "indices[" in steps,
            "calls": list(dict.fromkeys(c for c in calls if c)),
            "missing": miss_a or miss_b}


def compare(a: Rec, b: Rec, na: dict, nb: dict, fwd: dict, back: dict) -> list[dict]:
    """The exact changes between two paired records (one dict per changed slot)."""
    import tvm_ffi
    if a.kind != b.kind:
        return [{"slot": "stmt", "path": "stmt", "old_node": a.kind, "new_node": b.kind, "old_text": a.text,
                 "new_text": b.text, "ctx": [], "calls": [], "in_index": False, "dtype": None}]
    if a.kind == "Else":
        return []
    sa, sb = _slots(a), _slots(b)
    rec_op = a.feats.get("op")
    out = []
    for name in list(dict.fromkeys(list(sa) + list(sb))):
        x, y = sa.get(name, _MISSING), sb.get(name, _MISSING)
        if x is _MISSING or y is _MISSING:
            out.append({"slot": name, "path": name, "old_node": "missing" if x is _MISSING else _describe(x, na)[0],
                        "new_node": "missing" if y is _MISSING else _describe(y, nb)[0],
                        "old_text": "" if x is _MISSING else _describe(x, na)[1],
                        "new_text": "" if y is _MISSING else _describe(y, nb)[1],
                        "ctx": [], "calls": [rec_op] if rec_op else [], "in_index": False, "dtype": None,
                        "missing": True})
            continue
        if _is_obj(x) and _is_obj(y):
            try:
                mm = tvm_ffi.get_first_structural_mismatch(x, y, map_free_vars=True)
            except Exception:  # noqa: BLE001
                mm = None if _short(x, na) == _short(y, nb) else "text"
            if mm == "text":
                out.append({"slot": name, "path": name, "old_node": _tname(x), "new_node": _tname(y),
                            "old_text": _short(x, na), "new_text": _short(y, nb), "ctx": [_tname(x)],
                            "calls": [rec_op] if rec_op else [], "in_index": False, "dtype": _dtype(x)})
            elif mm is not None:
                out.append(_detail(name, x, y, mm, na, nb, rec_op))
        elif _is_obj(x) != _is_obj(y) or x != y:
            out.append({"slot": name, "path": name, "old_node": _describe(x, na)[0], "new_node": _describe(y, nb)[0],
                        "old_text": _describe(x, na)[1], "new_text": _describe(y, nb)[1], "ctx": [],
                        "calls": [rec_op] if rec_op else [], "in_index": False, "dtype": None})
    if not out:
        d = _operand_diff(a, b, na, nb, dict(fwd), dict(back))
        if d is not None:
            ox, oy = d
            out.append({"slot": "operand", "path": "operand", "old_node": "Var", "new_node": "Var",
                        "old_text": ox, "new_text": oy, "old_ty": na.get(ox, {}).get("ty"),
                        "new_ty": nb.get(oy, {}).get("ty"), "ctx": [], "calls": [rec_op] if rec_op else [],
                        "in_index": _in_brackets(a.text, ox), "dtype": None, "operand": True})
    return out


_MISSING = object()


def _in_brackets(text: str, tok: str) -> bool:
    """Is the first use of tok inside [...] (an index), rather than the buffer before it?"""
    depth = 0
    for m in _TOKEN.finditer(text):
        t = m.group(0)
        if t == "[":
            depth += 1
        elif t == "]":
            depth -= 1
        elif t == tok:
            return depth > 0
    return False


def _pair(olds: list, news: list) -> list[tuple[int, int]]:
    """Pair old/new records of a replaced block: positionally when the kinds line up, else greedily by kind and
    text similarity."""
    if len(olds) == len(news) and all(o.kind == n.kind for o, n in zip(olds, news)):
        return list(zip(range(len(olds)), range(len(news))))
    pairs, used = [], set()
    for j, n in enumerate(news):
        best, best_r = None, 0.3
        for i, o in enumerate(olds):
            if i in used or o.kind != n.kind:
                continue
            r = difflib.SequenceMatcher(a=o.key, b=n.key, autojunk=False).ratio() + (0.2 if o.line == n.line else 0)
            if r > best_r:
                best, best_r = i, r
        if best is not None:
            used.add(best)
            pairs.append((best, j))
    return sorted(pairs)


def diff_records(a: Flat, b: Flat) -> list[dict]:
    """Hunks between two flattened Functions (see the module docstring). Each hunk is JSON-ready:
    {tag: mod|del|add, old: [rec], new: [rec], pairs: [[i, j]], changes: [change with "pair"]}."""
    ka, kb = [_align_key(r) for r in a.recs], [_align_key(r) for r in b.recs]
    fwd = {p: p for p in a.params if p in b.params}
    back = {p: p for p in fwd}
    raw = []                                        # [tag, olds, news, pairs, changes]
    sm = difflib.SequenceMatcher(a=ka, b=kb, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                ra, rb = a.recs[i], b.recs[j]
                _bind_defs(ra, rb, fwd, back)
                if _operand_diff(ra, rb, a.names, b.names, fwd, back) is not None:
                    raw.append(["mod", [ra], [rb], [(0, 0)], compare(ra, rb, a.names, b.names, fwd, back)])
            continue
        olds, news = a.recs[i1:i2], b.recs[j1:j2]
        pairs = _pair(olds, news) if tag == "replace" else []
        for i, j in pairs:
            _bind_defs(olds[i], news[j], fwd, back)
        changes = []
        for i, j in pairs:
            changes += [{**c, "pair": [i, j]} for c in compare(olds[i], news[j], a.names, b.names, fwd, back)]
        raw.append([{"replace": "mod", "delete": "del", "insert": "add"}[tag], list(olds), list(news), pairs, changes])
    raw = _pair_shifted(raw, a, b, fwd, back)
    out = []
    for tag, olds, news, pairs, changes in raw:
        if tag == "mod" and pairs == [(0, 0)] and changes and "pair" not in changes[0]:
            changes = [{**c, "pair": [0, 0]} for c in changes]
        out.append({"tag": tag, "old": [r.to_dict() for r in olds], "new": [r.to_dict() for r in news],
                    "pairs": [list(p) for p in pairs], "changes": changes})
    return out


def _is_sync_rec(r: Rec) -> bool:
    from .classify_ir import SYNC_CLASSES, op_class
    return any(op_class(o) in SYNC_CLASSES for o in [r.feats.get("op")] + list(r.feats.get("ops", [])) if o)


def _pair_shifted(raw: list, a: Flat, b: Flat, fwd: dict, back: dict) -> list:
    """An edit inside an unrolled loop often shows as a SHIFT: `for w in range(4): ld(red[w])` -> `ld(red[w + 1])`
    aligns ld(red[1..3]) as equal and reports "ld(red[0]) deleted, ld(red[4]) added". A deleted and an added record
    from the same source line, kind and op are one source statement, so they are paired and diffed exactly (here:
    an index change). A same-line pair with identical text is a pure iteration shift and is dropped, unless it is a
    barrier / shuffle / async / atomic op (a moved sync is a sync change)."""
    dels = [(h, k) for h, item in enumerate(raw) if item[0] == "del" for k in range(len(item[1]))]
    adds = [(h, k) for h, item in enumerate(raw) if item[0] == "add" for k in range(len(item[2]))]
    sig = lambda r: (r.line, r.kind, r.feats.get("op"), r.depth) if r.line is not None else None  # noqa: E731
    used_add, drop_old, drop_new, extra = set(), set(), set(), []
    for h, k in dels:
        r = raw[h][1][k]
        s_r = sig(r)
        if s_r is None:
            continue
        for h2, k2 in adds:
            if (h2, k2) in used_add or sig(raw[h2][2][k2]) != s_r:
                continue
            s = raw[h2][2][k2]
            used_add.add((h2, k2))
            same = _operand_diff(r, s, a.names, b.names, dict(fwd), dict(back)) is None
            if same and _is_sync_rec(r):
                used_add.discard((h2, k2))           # a moved barrier stays a delete + an add
                break
            drop_old.add((h, k))
            drop_new.add((h2, k2))
            if not same:
                extra.append((h, ["mod", [r], [s], [(0, 0)], compare(r, s, a.names, b.names, fwd, back)]))
            break
    if not drop_old:
        return raw
    out = []
    for h, item in enumerate(raw):
        tag, olds, news = item[0], item[1], item[2]
        if tag == "del":
            olds = [r for k, r in enumerate(olds) if (h, k) not in drop_old]
        if tag == "add":
            news = [r for k, r in enumerate(news) if (h, k) not in drop_new]
        if olds or news:
            out.append([tag, olds, news, item[3], item[4]])
        out += [e for hh, e in extra if hh == h]
    return out


# ---- merge across shapes / repeats ---------------------------------------------------------------------------
def _sig(c: dict) -> tuple:
    return (c.get("slot"), c.get("path"), c.get("old_node"), c.get("new_node"),
            c.get("old_text") if c.get("operand") else None)


def merge(per_shape: list[tuple[str, list[dict]]]) -> list[dict]:
    """[(shape label, hunks)] -> merged hunks, grouped by (tag, old lines, new lines, kinds); repeats collapse."""
    groups: dict[tuple, dict] = {}
    for label, hunks in per_shape:
        for h in hunks:
            key = (h["tag"], tuple(r["line"] for r in h["old"]), tuple(r["line"] for r in h["new"]),
                   tuple(r["kind"] for r in h["old"]), tuple(r["kind"] for r in h["new"]))
            g = groups.get(key)
            if g is None:
                g = groups[key] = {"tag": h["tag"], "old": h["old"], "new": h["new"], "pairs": h["pairs"],
                                   "changes": [], "per_shape": {}, "callers": []}
            g["per_shape"][label] = g["per_shape"].get(label, 0) + 1
            for r in h["old"] + h["new"]:
                if r.get("caller") and r["caller"] not in g["callers"]:
                    g["callers"].append(r["caller"])
            seen = {_sig(c) for c in g["changes"]}
            for c in h["changes"]:
                if _sig(c) not in seen and len(g["changes"]) < 12:
                    seen.add(_sig(c))
                    g["changes"].append({**c, "shape": label})
    out = []
    all_shapes = [label for label, _ in per_shape]
    for g in groups.values():
        g["count"] = max(g["per_shape"].values())
        g["shapes"] = list(g["per_shape"])
        g["all_shapes"] = all_shapes
        g["callers"] = sorted(g["callers"])
        out.append(g)
    first = lambda g: min([r["line"] or 0 for r in g["old"] + g["new"]] or [0])  # noqa: E731
    return sorted(out, key=first)


def annotation(g: dict, n_shapes: int | None = None) -> str:
    parts = []
    if g.get("count", 1) > 1:
        parts.append(f"×{g['count']}")
    shapes = g.get("shapes", [])
    if n_shapes and len(shapes) < n_shapes:
        absent = [s for s in g.get("all_shapes", []) if s not in shapes]
        if absent and len(absent) < len(shapes):
            parts.append("not at " + "; ".join(absent))
        else:
            parts.append("only at " + "; ".join(shapes))
    if g.get("callers"):
        parts.append("called from " + ",".join(f"L{c}" for c in g["callers"]))
    return f"  [IR {', '.join(parts)}]" if parts else ""


def to_hunks(merged: list[dict], n_shapes: int | None = None) -> list[Hunk]:
    """Merged IR hunks -> diff.Hunk (kind, [(source line, canonical IR text)]); the first entry of each side
    carries the repeat count, the shapes and the call sites."""
    out = []
    for g in merged:
        note = annotation(g, n_shapes)
        old = [(r["line"] or 0, r["text"] + (note if k == 0 else "")) for k, r in enumerate(g["old"])]
        new = [(r["line"] or 0, r["text"] + (note if k == 0 and not old else "")) for k, r in enumerate(g["new"])]
        out.append(Hunk(g["tag"], old, new))
    return out


def shape_label(shape: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in shape.items())
