"""Classifier robustness probe (2026-11-16): label hand-made edits of softmax kernels and real GLM revisions.
Run: source env.sh && python scripts/experiments/classifier_probe.py   (CPU only, seconds). Results: notes/build_log.md section 8."""
import re, glob, os
from pathlib import Path
from akc.validation.changeaware import plan

def show(name, old, new, expect):
    p = plan("softmax", old, new)
    rules = sorted({f"{l.category}:{l.rule}" for l in p.labels})
    Ns = sorted({t.shape.get("N") for t in p.tests})
    dists = sorted({t.dist for t in p.tests})
    print(f"{name:34s} expect={expect:22s} got={','.join(p.cats) or '(none)':40s} tests={len(p.tests):2d} N={Ns} dists={dists}")
    print(f"{'':34s} rules={rules}")

opt = Path("workloads/softmax/cuda/softmax_opt.cu").read_text()
def mut(src, a, b, count=1):
    assert a in src, a
    return src.replace(a, b, count)

print("=== synthetic edits of softmax_opt.cu ===")
show("M1 rename m->row_max (no-op)", opt, re.sub(r"\bm\b", "row_max", opt), "(nothing)")
show("M2 reformat only (no-op)", opt, opt.replace("; ", ";\n    ").replace("{", "{\n"), "(nothing)")
show("M3 stride s>>=1 -> s>>=2", opt, mut(opt, "s >>= 1) {\n    if (tid < s) red[tid] +=", "s >>= 2) {\n    if (tid < s) red[tid] +="), "sync/indexing")
show("M4 += -> = in sum reduction", opt, mut(opt, "red[tid] += red[tid + s]", "red[tid] = red[tid + s]"), "numerical/logic")
show("M5 red[0] -> red[1]", opt, mut(opt, "1.0f / red[0]", "1.0f / red[1]"), "indexing")
show("M6 * inv -> * inv * inv", opt, mut(opt, "* inv;", "* inv * inv;"), "numerical/logic")
show("M7 BLOCK/2 -> BLOCK/4 (sum red.)", opt, opt[:opt.index("// 2)")] + mut(opt[opt.index("// 2)"):], "BLOCK / 2", "BLOCK / 4"), "tiling/sync")
show("M8 remove a __syncthreads", opt, mut(opt, "  m = red[0];\n  __syncthreads();", "  m = red[0];"), "sync")
show("M9 -INFINITY -> 0.0f", opt, mut(opt, "-INFINITY", "0.0f"), "numerical")
show("M10 BLOCK 128 -> 256", opt, mut(opt, "#define BLOCK 128", "#define BLOCK 256"), "tiling")
show("M11 tid -> tid+1 in loop start", opt, mut(opt, "for (int i = tid; i < N; i += BLOCK) sum", "for (int i = tid + 1; i < N; i += BLOCK) sum"), "indexing")

g3 = Path("results/pilot/20261115-203841/softmax_glm_r3.cu").read_text()
good = mut(g3, "l = (e0 + e1) + (e2 + l);", "l = l + ((e0 + e1) + (e2 + e3));")
show("G1 GLM dropped term (real bug)", good, g3, "numerical/logic")
show("G2 shfl_xor -> shfl_down", g3, mut(g3, "__shfl_xor_sync(0xffffffffu, m, off)", "__shfl_down_sync(0xffffffffu, m, off)"), "sync")
show("G3 shfl offset 16 -> 8", g3, mut(g3, "for (int off = 16; off > 0; off >>= 1)\n        m =", "for (int off = 8; off > 0; off >>= 1)\n        m ="), "sync/indexing")
show("G4 unroll guard 384 -> 383", g3, mut(g3, "while (N - i > 384)", "while (N - i > 383)"), "indexing")

print("\n=== real GLM revisions (pilot) ===")
ref = Path("workloads/softmax/cuda/softmax_ref.cu").read_text()
for d in sorted(glob.glob("results/pilot/*/")):
    files = sorted(glob.glob(d + "softmax_glm_r*.cu")) or sorted(glob.glob(d + "softmax_txl_r*.py"))
    for i, f in enumerate(files):
        old = Path(files[i-1]).read_text() if i else None
        if old is None:
            continue
        new = Path(f).read_text()
        p = plan("softmax", old, new)
        nchg = sum(max(len(h.old), len(h.new)) for h in p.hunks)
        print(f"{d.split('/')[-2]} {Path(files[i-1]).stem[-2:]}->{Path(f).stem[-2:]}  changed lines={nchg:3d}  cats={','.join(p.cats) or '(none)'}  tests={len(p.tests)}")
