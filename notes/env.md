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

### CAKE availability (big risk)
- No official CAKE repo found (searched arXiv page, GitHub, author pages). A third-party blog post
  (gangliao.me, Oct 2026) also says no release exists.
- An unofficial re-implementation `qhy991/open-cake-ir` showed up in search but is now 404 / private.
- → We need a team decision: (a) email the authors, (b) build a minimal "CAKE-like" IR + lowering ourselves,
  or (c) swap the baseline (e.g. Triton/CUDA source written by the agent).
