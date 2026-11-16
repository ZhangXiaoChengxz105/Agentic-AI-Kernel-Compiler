# Agentic-AI-Kernel-Compiler

CS 521 project (Hongyi Chen, Minghao Fang, Xiaocheng Zhang). An LLM agent writes GPU kernels; we check every
revision with **VOLTA** formal verification and **change-aware tests** generated from what the revision changed,
and turn the findings into feedback for the next revision. Compared configurations: (A) fixed tests,
(B) fixed tests + VOLTA, (C) fixed tests + VOLTA + change-aware tests.

**Architecture and module contracts: [`docs/architecture.md`](docs/architecture.md).**

## Setup (NCSA Delta)
```bash
bash setup_env.sh --tests   # builds everything inside this folder: Rust, libz3, VOLTA, Python env (envs/akc), checks
source env.sh               # every new shell: paths, caches, activates envs/akc
```
Details and troubleshooting: [`docs/setup.md`](docs/setup.md). The Lumen key goes in `.secrets/lumen.env` (never committed).

## Try it
```bash
akc hottest --workload softmax --new workloads/softmax/cuda/bugs/softmax_bug_tail.cu \
            --old workloads/softmax/cuda/softmax_opt.cu          # what changed + targeted checks + problems
akc validate --workload gemm --kernel my_gemm.cu --config C       # the feedback the agent would receive
akc eval-bugs                                                     # A/B/C on the injected bugs
python -m unittest discover -s tests                              # 86 tests (GPU tests skip on login nodes)
sbatch scripts/slurm/smoke.sbatch                                 # full GPU run (uses the allocation: ask first)
```

## Layout
```
akc/            the system (task, llm, agent, backend, validation, feedback, evaluation)
workloads/      one folder per operator: manifest.json + cuda/ (tirx/) kernels and injected bugs
scripts/slurm/  GPU jobs          tests/   unit + integration tests (mirrors akc/)
docs/           architecture, setup          notes/   dated decisions and experiment logs
setup/          exact package lock           results/ experiment outputs (JSON)
third_party/    VOLTA, TIRx-harness, tirx-kernels, Lumen (fetched by setup_env.sh, not committed)
```

## Status (2026-11-16)
Tasks 1–2 done (environment, baseline harness, change-aware testing); VOLTA integrated as a validator; GLM-5.3
writes correct softmax kernels in CUDA and tirx-lite; first A100 run: injected bugs detected A 4/11, B 5/11, C 11/11.
See `notes/gpu_results.md`, `notes/glm_pilot.md`, `notes/build_log.md`.
