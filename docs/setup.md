# Environment setup

Everything lives inside the repository folder (CLAUDE.md rule 1): no changes to `$HOME`, no `sudo`, no
`pip --user`, no conda base. One script builds it all; it is idempotent (finished steps are skipped) and logs
to `logs/setup_<timestamp>.log`.

```bash
git clone git@github.com:ZhangXiaoChengxz105/Agentic-AI-Kernel-Compiler.git && cd Agentic-AI-Kernel-Compiler
bash setup_env.sh --tests      # ~9 min from scratch on Delta (verified 2026-11-16 in a fresh clone)
source env.sh                  # in every new shell
bash setup_env.sh --check      # verify later
```

## What gets installed
| step | location | notes |
|---|---|---|
| Rust stable + 1.95.0 | `tools/rustup`, `tools/cargo` | stable builds VOLTA (edition 2024); 1.95.0 for TIRx NumSim (cargo 1.100's build-dir layout breaks tirx-harness) |
| libz3 5.4.1 | `tools/z3` | from pip `z3-solver`; VOLTA links it (`env.sh` sets `LIBRARY_PATH`/`LD_LIBRARY_PATH`) |
| VOLTA | `third_party/volta` @ c091559 | `cargo build --release -j 4` |
| TIRx-harness, tirx-kernels | `third_party/` @ 99522bf, 6d2f3e2 | sources for docs/examples (the packages come from the lock) |
| Lumen | `third_party/lumen` @ 42c8822 | API docs only |
| Python 3.13 env | `envs/akc` | exact versions from `setup/requirements-akc.lock`, then `pip install -e .` |

Key versions: torch 2.14.1+cu132, apache-tvm-tirx 0.1.0rc2, tirx-harness 0.1.3rc0, tirx-kernels 0.1.3rc1,
cuda-bindings 13.5.0, numpy 2.5.5. Delta: A100-SXM4-40GB nodes, driver 595.71.05 (CUDA 13.2), nvcc 13.2.

## Known pitfalls (already handled by the script)
- **apache-tvm-tirx 0.1.0rc2 is gone from the PyPI index** but tirx-harness 0.1.3rc0 requires it: the lock
  installs it from its original `files.pythonhosted.org` URL with its sha256.
- **torch and tirx-harness pin different NVRTC patch versions** (13.2.86 vs 13.2.78): the lock is installed
  with `--no-deps`, keeping 13.2.86; TIRx compile and Racecheck were re-verified with it.
- **`module load` inside a pipe does nothing** (it runs in a subshell): the script calls it directly, or set
  `PYTHON313=/path/to/python3.13`.
- **VOLTA needs libz3 at run time**: always `source env.sh` (akc's VOLTA runner also sets it for its subprocess).

## GPU and Slurm
- Login nodes have no GPU: GPU tests report "skipped / not verified"; VOLTA and NumSim run on the CPU.
- On 2026-11-15 the batch partition `gpuA100x4` had a ~13-day queue; `gpuA100x4-interactive` (≤ 1 h) started
  within minutes, so `scripts/slurm/*.sbatch` use it. Account: `bidr-delta-gpu`.
- Keep heavy CPU work (large VOLTA runs, NumSim sweeps) off the login node.

## Secrets (never committed; `.secrets/` is git-ignored)
- Lumen: `echo 'export LUMEN_API_KEY=sk_...' > .secrets/lumen.env && chmod 600 .secrets/lumen.env`
- Git over SSH (optional): key in `.secrets/ssh/id_ed25519`; `env.sh` points `GIT_SSH_COMMAND` at it and keeps
  git's global config in `./.gitconfig` (`GIT_CONFIG_GLOBAL`).
