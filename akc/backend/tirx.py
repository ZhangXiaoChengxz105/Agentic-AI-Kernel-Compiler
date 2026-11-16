"""tirx-lite helpers: safety scan for LLM-written modules, loading, and sm_80 compilation.

A tirx-lite workload module exposes a factory (manifest "tirx.factory_args", e.g. make_kernel(rows, n))
that returns a @txl.kernel whose tensor parameters are listed in manifest "tirx.tensors" (inputs first,
output last). Modules written by an LLM are executed only after safety_scan() finds nothing.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ALLOWED_IMPORTS = {"tirx_kernels.tirx_lite", "tirx_kernels", "math", "typing", "functools"}
FORBIDDEN_NAMES = {"os", "sys", "subprocess", "shutil", "socket", "open", "eval", "exec", "compile", "__import__",
                   "importlib", "pathlib", "builtins", "globals", "locals", "setattr", "delattr", "input", "breakpoint"}


def safety_scan(src: str) -> list[str]:
    """Reasons the code must NOT be executed (empty list = ok to import)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [f"import of {a.name} not allowed" for a in node.names if a.name not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and (node.module or "") not in ALLOWED_IMPORTS:
            problems.append(f"import from {node.module} not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute {node.attr} not allowed")
    return problems


def load_module(path: str | Path, name: str = "candidate_txl"):
    problems = safety_scan(Path(path).read_text())
    if problems:
        raise PermissionError("; ".join(problems))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_kernel(mod, workload_meta: dict, shape: dict):
    """Call the module's factory with the shape values named by the manifest contract."""
    args = [shape[k] for k in workload_meta["tirx"]["factory_args"]]
    return mod.make_kernel(*args)


def compile_sm80(kernel):
    from tirx_kernels.runner import compile_kernel
    return compile_kernel(kernel.func, backend_config={"cuda": {"arch": "sm_80"}})
