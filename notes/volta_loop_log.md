# VOLTA test loop: build log (2026-11-15)

Goal: get an end-to-end, scriptable VOLTA check running on **our own** kernels before anything else
(CPU only, no GPU, no Lumen). Operator: row-wise softmax. Status: **done, 10/10 + 10/10 results as expected,
17/17 tests pass.**

## Pipeline
```
kernels/softmax/*.cu ──(B1 verify.ptx, mode=volta)──▶ build/ptx/softmax/*.volta.ptx
                                                          │
kernels/softmax/manifest.json ──(B3 verify.cases)──▶ VoltaCase (memory layout, params, launch dims)
                                                          │
                                       (B2 verify.volta.run_compare)
                                                          ▼
                                    VoltaResult {status, which, detail, timing, ...}
                                                          │
                                 (B4 scripts/run_volta_softmax.py) ──▶ table + results/volta/*.json
```

## Building blocks

### B0. Kernels: `kernels/common.cuh`, `kernels/softmax/*.cu`, `manifest.json`
- **Contract for every kernel**: `extern "C" __global__` (unmangled names), write `EXP(x)`, never `expf` directly;
  compile with `-I kernels/`.
- `EXP(x)` = `__symexpf(x)` (body-less extern → stays `call __symexpf` in PTX, which VOLTA reads as exact exp)
  when `-DVOLTA`, else `expf(x)` for real GPU builds.
- softmax signature `(const float* x, float* y, int N)`, row-major `[rows, N]`, one CTA per row.

| Variant | Block | Change vs opt | Bug category | Expected VOLTA |
|---|---|---|---|---|
| softmax_ref | 1 | (serial reference) | — | — |
| softmax_opt | 128 | (tree reductions, ×1/sum) | — | EQUIV |
| softmax_bug_nosync | 128 | `__syncthreads()` removed in max reduction | sync | RACE |
| softmax_bug_tail | 128 | reductions stop at `(N/128)*128` | tiling | EQUIV if N%128==0 else NOT_EQUIV |
| softmax_bug_offbyone | 128 | sum loop `i <= N` | indexing | NOT_EQUIV |
| softmax_bug_nomax | 128 | no max subtraction | numerical | EQUIV (blind spot) |

`manifest.json` stores name, file, role, block size, bug {category, change}, machine-readable `expect_volta`.

### B1. `verify/ptx.py` → `compile_to_ptx(source, out_dir, mode="volta"|"gpu", arch="sm_80")`
| | |
|---|---|
| Input | `.cu` path, output dir, mode, arch, extra include dirs / defines |
| Output | `PtxBuild(source, ptx, mode, arch, kernels, cmd, seconds, warnings)`; `kernels` = `.entry` names parsed from PTX |
| Errors | `CompileError(cmd, stderr)`: stderr can be fed to the agent as-is |
| Conditions | `nvcc` on PATH. mode=volta adds `-DVOLTA` and **no** `-lineinfo`; mode=gpu adds `-lineinfo` |
| Cost | ~0.6 s per file |

### B2. `verify/volta.py`
| Piece | Input | Output |
|---|---|---|
| `ArraySpec(name, base, elem_bytes, length, kind)` | kind ∈ in/out/inout/index | `--array name:base:width:len:kind` |
| `VoltaCase(...)` | ref/cand PTX + kernel names, ref/cand block, grid, arrays, params (in declaration order: `ptr:x`, `int:N`, …), check_arrays, timeout | `validate()` rejects overlapping arrays, `ptr:` to unknown arrays, checking a non-output array |
| `build_command(case)` | VoltaCase | argv list (`--block2` only if blocks differ) |
| `run_compare(case)` | VoltaCase | `VoltaResult`; sets `LD_LIBRARY_PATH` to `tools/z3/z3/lib` itself |
| `parse_output(label, exit, stdout, stderr)` | raw VOLTA output | `VoltaResult` (pure, unit-tested) |

`VoltaResult` fields: `status, which (reference|candidate), detail, mismatches, mismatch_examples,
elements_checked, instructions, block_syncs, exec_s, vc_s, wall_s, exit_code, cmd, stdout, stderr`.

| status | meaning | VOLTA text (source: volta_cli/main.rs, eval/error.rs) |
|---|---|---|
| EQUIV | all checked elements proved equal over ℝ | `EQUIVALENT`, exit 0 |
| NOT_EQUIV | some elements differ | `NOT EQUIVALENT: K mismatched element(s)` + `y[i]` list |
| RACE | data race | `data race on Shared[0x4]: write by t1 at pc:64 conflicts with read by t0 at pc:98` |
| DEADLOCK | threads stuck at barrier | `deadlock: K thread(s) blocked` |
| OUT_OF_BOUNDS | access outside declared arrays | `out-of-bounds ... access at 0x...` |
| UNINIT | uninitialized value reaches output | `uninitialized` / `is undefined (uninitialized read)` |
| FOOTPRINT | kernels wrote different element sets | `written footprints differ` |
| UNSUPPORTED | outside VOLTA scope | `not a structured-CTA`, `unsupported at`, `lowering failed`, `instruction limit exceeded`, … |
| CONFIG_ERROR | our setup wrong | `configuration error`, `no kernel named`, CLI exit 2 |
| TIMEOUT / ERROR | wall-clock limit / unrecognized output | — |

What EQUIV does **not** mean: float rounding/overflow/NaN are not modeled; only CTA 0 runs; only the given N.

### B3. `verify/cases.py` → `softmax_case(ref_ptx, ref, cand_ptx, cand, N)`
| | |
|---|---|
| Input | the two PTX files, their manifest entries, row length N |
| Output | `VoltaCase` labelled `"<ref> vs <cand> @N=<N>"` |
| Memory layout | `x`: in, f32, 2·N elems @0x100000 (2 rows so an off-by-one reads real neighbour data); `y`: out, f32, 2·N @0x8000000; params `(ptr:x, ptr:y, int:N)`; grid (2,); ref block 1, cand block 128 |
| Conditions | N ≥ 1 and the layout must not overlap (N < ~16M) |

### B4. `scripts/run_volta_softmax.py [--sizes 256,200] [--timeout 600]`
- Compiles every manifest variant (B1), runs ref vs every candidate for every N (B3 → B2), compares with
  `expect_volta`, prints a table, saves `results/volta/softmax_<timestamp>.json` (tool versions, compile
  commands, full results incl. raw output). Exit 0 iff everything matched expectations.

### B5. `tests/test_volta.py` (stdlib unittest)
`source env.sh && envs/tirx/bin/python -m unittest discover -s tests -v` → 17 tests, ~5 s.
Parser tests use real captured VOLTA output; integration tests compile + run VOLTA, including a kernel with a
data-dependent branch (→ UNSUPPORTED: "guard predicate is symbolic … not a structured-CTA"), a compile error,
the shape-dependent tail bug, and the timeout path.

## Results
| N | opt | bug_nosync | bug_tail | bug_offbyone | bug_nomax |
|---|---|---|---|---|---|
| 256 (÷128) | EQUIV 0.06 s | RACE 0.02 s | **EQUIV (bug hidden)** 0.06 s | NOT_EQUIV 0.15 s | **EQUIV (blind spot)** 3.3 s |
| 200 | EQUIV 0.04 s | RACE 0.02 s | NOT_EQUIV 0.10 s | NOT_EQUIV 0.09 s | **EQUIV (blind spot)** 1.3 s |
| 1024 (÷128) | EQUIV 0.51 s | RACE 0.03 s | **EQUIV (bug hidden)** 0.53 s | NOT_EQUIV 2.4 s | **EQUIV** 209.7 s |
| 1000 | EQUIV 0.52 s | RACE 0.03 s | NOT_EQUIV 2.7 s | NOT_EQUIV 2.3 s | **EQUIV** 157.4 s |

Whole 1024/1000 run: 6 min 20 s wall, 731 MB peak RSS (single core, login node).
Raw: `results/volta/softmax_20261115-184345.json`, `results/volta/softmax_20261115-185108.json`.

## Problems hit and how they were solved
| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `libz3.so.5.4: cannot open shared object file` | VOLTA binary needs `tools/z3/z3/lib` on `LD_LIBRARY_PATH`; only set by `env.sh` | `run_compare` sets it in the subprocess env itself |
| 2 | VOLTA would not see `expf` as exp | nvcc expands `expf` into `ex2.approx` + range reduction | `EXP()` macro → `__symexpf` under `-DVOLTA`; dual compile (VOLTA build vs GPU build) |
| 3 | bug variants failed: `../common.cuh: No such file` | relative include from a deeper dir | `#include "common.cuh"` + `-I kernels/` |
| 4 | `lowering failed: Undefined label $L__BB0_21` | `-lineinfo` puts `.loc` right after labels; VOLTA's lowering drops such labels (`volta parse` still passes). Confirmed: stripping only `.loc` lines fixes it | VOLTA PTX is built **without** `-lineinfo` (candidate upstream bug report) |
| 5 | (design) ref and opt must write the same elements in CTA 0 | VOLTA pairs CTA-0 footprints | both kernels map one CTA per row; ref uses `-b 1`, opt `--block2 128` |
| 6 | (observation) bug_nomax proof cost explodes: 3.3 s @256 → 210 s @1024 | proving shift invariance of exp/Σexp is expensive for the decision procedure | keep VOLTA sizes small; always set a timeout; overhead must be reported per config |

## Takeaways for the project
- VOLTA works on our softmax and gives the right verdict for race, indexing and (shape-permitting) tiling bugs.
- Two concrete gaps, shown by experiment, are exactly what change-aware tests (Task 2) must cover:
  1. **shape dependence**: `bug_tail` is proved "equivalent" at N=256/1024. VOLTA only proves the given N.
  2. **real-number blind spot**: `bug_nomax` is proved equivalent at every N, yet overflows in float32.
- VOLTA cost varies by ~400× between kernels of the same size: an overhead metric is needed.

## Next
GEMM and LayerNorm through the same loop (1.2), then env + PyTorch (1.3) and the GPU harness (1.4/1.5), which
will actually show the nomax overflow and the tail bug at runtime.
