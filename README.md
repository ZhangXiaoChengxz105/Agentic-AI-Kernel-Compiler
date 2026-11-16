# Agentic-AI-Kernel-Compiler

CS 521 project: validate agent-written GPU kernels with **VOLTA** (formal equivalence/race checking of PTX)
plus **change-aware tests** (targeted tests derived from what each revision changed). See `CLAUDE.md` for the
project brief and `notes/` for decisions and logs (`notes/build_log.md` is the current overview).

## Setup (NCSA Delta)
```bash
source env.sh                      # all tools/caches/keys stay inside this folder
source envs/tirx/bin/activate      # Python 3.13 + torch 2.14.1+cu132 + TIRx
```
Third-party sources live in `third_party/` (git-ignored): VOLTA (built with `cargo build --release`),
TIRx-harness, tirx-kernels, Lumen.

## Common commands
```bash
python -m unittest discover -s tests                          # 64 tests (3 need a GPU)
python scripts/run_volta.py --op softmax|gemm|layernorm       # VOLTA on ref vs every variant
python scripts/run_change_aware.py                            # Task 2 on all injected bugs (CPU)
python -m harness.cli check --op gemm                         # GPU: fixed tests on every variant
python -m harness.cli bench --op gemm                         # GPU: timing vs PyTorch
python -m agent.lumen usage | models | chat --dry-run "..."   # Lumen (chat costs coins)
sbatch scripts/slurm/smoke.sbatch                             # GPU smoke test (uses the allocation)
```

## Layout
`kernels/` CUDA kernels + injected bugs · `verify/` PTX build + VOLTA · `ref/` PyTorch references ·
`harness/` GPU correctness + benchmark · `changeaware/` Task 2 · `agent/` Lumen client (agent loop later) ·
`scripts/` drivers + Slurm · `tests/` · `results/` · `notes/`
