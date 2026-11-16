# Environment notes (Task 1.1)

Last updated: 2026-11-15

## Hardware / cluster
- Target GPU: NVIDIA A100 on NCSA Delta → compute capability **sm_80**.
- Login node: 128 cores, 250 GB RAM (shared). No GPU on login node; GPU work goes through Slurm.

## System toolchain (preinstalled, read-only)
| Tool | Version | Location |
|---|---|---|
| nvcc (CUDA) | 13.2 (V13.2.78), via `cudatoolkit/26.5_13.2` module (loaded by default) | `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/bin/nvcc` |
| Alt CUDA | `cudatoolkit/26.5_12.9-unsupported` module also exists | — |
| System Python | 3.9.21 (too old for current PyTorch; project env TBD in Task 1.3) | `/usr/bin/python3` |

CUDA 13 still supports sm_80 (it dropped only sm_7.0 and older), so A100 is fine.
The GPU driver version has to be checked on a compute node with `nvidia-smi` (needs Slurm, so ask first).

## Workspace-installed tools (all under `$CS521/tools`, wired up in `env.sh`)
| Tool | Version | Why |
|---|---|---|
| Rust (rustup/cargo) | rustc 1.100.0 | VOLTA is written in Rust (edition 2024 → needs ≥1.85) |
| libz3 | 5.4.1 (from pip `z3-solver`, `tools/z3/z3/lib`) | VOLTA links against libz3 |

## Third-party sources (`third_party/`, git-ignored)
| Project | Source | Commit | Status |
|---|---|---|---|
| VOLTA | https://github.com/willtunnels/volta (MIT) | `c091559` (2026-10-28) | Built (`cargo build --release`, ~3 min). Smoke tests pass (see volta_scope.md) |
| CAKE | Paper arXiv:2608.12629 (NVIDIA/CMU, Aug 2026) | — | **No public code found.** See "CAKE availability" below |

### VOLTA facts
- Input: **PTX** (two kernels: reference + optimized). Runs on **CPU**, no GPU needed.
- Binaries: `third_party/volta/target/release/volta` (parse / analyze / compare), `volta-bench`.
- Kernel requirements: static array sizes, statically resolvable branches/addresses, no recursion.
  Sync: `bar.sync`/`syncwarp`/tensor-core implicit barriers only. **No async (`cp.async` pipelines, `arrive`, TMA, `wgmma`).**
- Check is over the **reals** (not IEEE floats), one CTA (CTA-0) at a time.

### CAKE availability (big risk), rechecked 2026-11-15
- **No official CAKE code.** The paper never promises a release; only the *generated CUDA kernels* were
  upstreamed (FlashInfer PRs #4262, #4274, #4279, #4287). Checked: arXiv HTML, GitHub search API,
  authors' accounts (yzh119, tqchen), flashinfer-ai, mlc-ai.
- Candidates for a substitute:
  | Repo | What it is | Notes |
  |---|---|---|
  | `huangzhilin-hzl/cake-ir` | Independent re-implementation (Python): typed schedule IR, verifier gates with stable error codes, CUDA lowering (SIMT/WMMA GEMM, elementwise), sm_80 target | 1 commit (2026-08-17), license "Other" → check before use. No softmax/LayerNorm |
  | `mlc-ai/TIRx-harness` | Open "compiler harness for agentic GPU programming" from the MLC/TVM group (Tianqi Chen is a CAKE co-author). Has sync/race/numerical analyses. Apache-2.0, pip `tirx-harness`, actively maintained | Blog calls CAKE a *concurrent* effort, not the same code. Experiments only on Blackwell; sm_80 support unknown |
  | `QiGS/cake-ascend` | Re-implementation targeting Huawei Ascend | Not usable on A100 |
  | `qhy991/open-cake-ir` | Unofficial re-implementation | Now 404 / private |
- → Team decision needed. Options: use TIRx-harness as the "CAKE-style" baseline (if it supports sm_80),
  use `cake-ir`, build our own minimal IR, or email the authors.

### TIRx-harness hands-on trial (2026-11-15)
Env: `envs/tirx` (Python 3.13.5 from module `python/3.13.5-gcc13.3.1`, venv in workspace, 1.3 GB).
Sources also cloned: `third_party/tirx-harness` (`99522bf`), `third_party/tirx-kernels`.

Install notes (upstream packaging is currently broken):
- PyPI default `tirx-harness 0.1.2.post2` lacks `tvm_tirx` → unusable.
- `tirx-harness 0.1.3rc0` / `tirx-kernels 0.1.3rc1` pin `apache-tvm-tirx==0.1.0rc2`, **which was removed from PyPI**.
  Workaround used: install both with `--no-deps`, plus `apache-tvm-tirx[cuda]==0.1.0` and the other deps by hand.
  `import tvm_tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness` → OK.

Results:
| Check | Result |
|---|---|
| Compile a TIRx-lite kernel with `arch=sm_80` | ✅ OK (generates CUDA C++, NVRTC-compiled). `nvidia/nvidia-a100 → sm_80` tag exists |
| Generated CUDA → `nvcc -arch=sm_80 -ptx` → `volta parse` | ✅ OK: full TIRx → PTX(sm_80) → VOLTA path works on a toy kernel |
| Bundled kernels (gemm/norm/attention…) on sm_80 | ❌ All target sm_90+/sm_100a; e.g. `norm/rmsnorm.py` uses `mul.rz.ftz.f32x2` (Blackwell-only). We would write our own GEMM/softmax/LayerNorm |
| `numsim.racecheck` (CPU race checker) | ❌ Fails for sm_80 *and* sm_100a: `target PTX operation "tirx.ptx.add_int_sm107f" has no reviewed registry row` → version mismatch from the rc2→0.1.0 substitution, not an arch problem (verified below) |

Root cause (verified): at startup NumSim sends the compiler's *entire* PTX op table
(`tvm_tirx/backend/cuda/ptx/table.py`, 545 entries in 0.1.0) to its Rust frontend, which requires a registry row
for every entry (`transpiler/native_frontend.py`: "Every target PTX operation requires an explicit Rust registry row").
harness 0.1.3rc0 has 507 rows (written against rc2). 39 entries added in tvm-tirx 0.1.0 have no row
(`add_int_sm107f`, `sub_int_sm107f`, `neg_int_sm107f`, `max/min_int8x4`, `fabric_*_tensor*`, `tcgen05_shift`, …).
`add_int_sm107f` is just the alphabetically first one. The check runs before looking at the kernel, so *every* kernel fails.
Timeline: harness 0.1.3rc0 uploaded 2026-11-12 (pins rc2); tvm-tirx 0.1.0 uploaded 2026-11-13 and rc2 removed.
→ Expect a matching harness release soon; re-check PyPI before building anything from source.

Takeaway: TIRx works as an A100 kernel-authoring/compile frontend, and it pipes into VOLTA.
Its own analyses are blocked until upstream republishes matching packages (or we build tvm-tirx from source, which is heavy, needs LLVM).

### Update (2026-11-15, later): TIRx-harness analyses now WORK on sm_80
Fix 1: found the deleted `apache-tvm-tirx==0.1.0rc2` wheel. Its original files.pythonhosted.org URL and sha256 are in
`third_party/tirx-harness/uv.lock`, and the file is still served. Downloaded to `tools/wheels/`, sha256 verified
(`62273d1c…989d2c`), installed into `envs/tirx`. `pip check` → no broken requirements.
Fix 2: NumSim builds its Rust engine on first use. With cargo 1.100 the rlibs land in `release/build/*/out/`
(new Cargo build-dir layout) but harness looks in `release/deps/` → "Cargo produced 0 'numsim_engine' rlib candidates".
Fixed by using **Rust 1.95.0** (the documented minimum) for TIRx only: `export RUSTUP_TOOLCHAIN=1.95.0`
(installed via rustup in `tools/rustup`; VOLTA keeps using the default 1.100).
First NumSim run compiles the engine (~5 min, cached in `.cache/tirx-numsim`), later runs take seconds.

racecheck results (arch=sm_80, 1 CTA × 64 threads):
| Kernel | Verdict |
|---|---|
| each thread writes own smem slot | `clean` ✅ |
| warp 0 and warp 1 both write smem[0..31], no barrier | `error`: `data_race`, `write_write`, `missing_inter_actor_sync`, points at the exact source line ✅ |
| 32 threads of ONE warp write smem[0] in one instruction | `clean`. Same-instruction intra-warp writes aren't treated as a race (same as compute-sanitizer) |

How to use TIRx env:
```bash
source env.sh && source envs/tirx/bin/activate && export RUSTUP_TOOLCHAIN=1.95.0
```

### Update (2026-11-15, evening): PyTorch, driver, quota, Slurm queue
- **Driver**: login node has NVIDIA driver packages **595.71.05** (`/usr/lib64/libcuda.so.595.71.05`) → CUDA 13.2 level.
  Compute-node driver still to be confirmed with `nvidia-smi` (scripts/slurm/smoke.sbatch records it).
- **PyTorch**: `torch 2.14.1+cu132` installed into `envs/tirx` (`pip install --no-cache-dir torch==2.14.1 --index-url
  https://download.pytorch.org/whl/cu132`), 2m55s; env now 6.2 GB. Chosen because cu132 matches driver 595 and 2.14.x is
  the series TIRx's own deps use (2.15.0 was newer and skipped).
  Known benign conflict: torch pins `nvidia-cuda-nvrtc 13.2.86` (via cuda-toolkit==13.2.2), tirx-harness pins `==13.2.78`.
  Kept 13.2.86; re-verified TIRx sm_80 compile + racecheck still work.
- **GPU kernel launching from Python**: `cuda-bindings 13.5` (already in env) driver API (`harness/launch.py`).
- **Quota**: /u/mfang 72.8 GB used of 100 GB soft (103 GB hard), 415k of 750k files → ~27 GB headroom. Avoid pip caches.
- **Slurm** (`sbatch --test-only`, nothing submitted): account `bidr-delta-gpu` valid (749 of 3000 h left).
  Estimated start for a 30-min 1-GPU job: `gpuA100x4` ≈ 13 days (2,730 pending jobs), `gpuA100x4-preempt` ≈ 2 days,
  `gpuA100x4-interactive` ≈ minutes (max 1 h) → scripts default to the interactive partition.
- **Login-node limits**: a VOLTA LayerNorm run at N≥512 reached ~10 GB RSS after 3.5 min and was stopped; run VOLTA
  scaling experiments on a compute node, keep login-node VOLTA runs small (layernorm N ≤ 200).
