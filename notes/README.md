# notes/: dated decision and experiment logs

These files record what was done and why, in order; they are not kept in sync with the code. Paths written
before the 2026-11-16 restructure use the old layout:

| old path (≤ 2026-11-15) | current location |
|---|---|
| `kernels/<op>/…`, `kernels/common.cuh` | `workloads/<op>/cuda/…`, `workloads/common/common.cuh` |
| `kernels_txl/softmax/…` | `workloads/softmax/tirx/…` |
| `verify/ptx.py` | `akc/backend/ptx.py` |
| `verify/volta.py`, `verify/cases.py`, `verify/hints.py` | `akc/validation/volta/{runner,layouts,hints}.py` |
| `ref/` | `akc/task/{reference,inputs}.py` |
| `harness/check.py`, `runner.py`, `worker.py` | `akc/validation/runners/{check,gpu,gpu_worker}.py` |
| `harness/launch.py`, `bench.py` | `akc/backend/{launch,bench}.py` |
| `harness/fixed_suite.py` | `akc/validation/fixed_tests/suite.py` (shapes now in manifests) |
| `harness/cli.py`, `scripts/run_*.py` | `akc` CLI (`akc/cli.py`, `akc/evaluation/*`) |
| `changeaware/` | `akc/validation/changeaware/` |
| `agent/lumen.py`, `agent/tirx_eval.py` | `akc/llm/lumen.py`, `akc/validation/runners/numsim.py` |
| `scripts/pilot_glm_softmax.py` | `akc/agent/pilot.py` (`akc pilot`) |
| `envs/tirx` | `envs/akc` (built by `setup_env.sh`) |

Index: `env.md` (environment history), `volta_scope.md`, `volta_loop_log.md`, `build_log.md`,
`glm_pilot.md`, `gpu_results.md`, `lumen_models.md`, `plan_tasks_1_2.md`.
Current architecture: `docs/architecture.md`.
