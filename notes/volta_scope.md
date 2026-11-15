# VOLTA feasibility (Task 1.2) — preliminary

Last updated: 2026-11-15. VOLTA commit `c091559`.

## Smoke tests run (login node, CPU, `volta-bench --iterations 1 single ...`)
| Benchmark | Result | Time |
|---|---|---|
| (MatMul-1, MatMul-2) | EQUIV, 4096/4096 output elements proved | VC gen 43 s + solve 89 s |
| Red-5 (racy) | RACE detected (Shared[0x4], t1 write vs t0 read) | <0.01 s |
| LayerNorm (pre-fix) | RACE detected (Shared[0x0]) | <0.01 s |

Note: equivalence checking is CPU-heavy (~2 min for a 64×64 matmul). Larger runs should go to a compute node, not the login node.

## Per-kernel outlook (from README + source; still to be confirmed with our own kernels)
| Kernel | Outlook | Why |
|---|---|---|
| Tiled GEMM | **Supported** | Polynomial fragment; bundled MatMul/GEMM/TileLang benchmarks prove equivalence. Must avoid `cp.async`/TMA pipelines. |
| Row-wise softmax | **Partially**: needs testing | `exp`/`max` fragment is handled (attention benchmarks), but division/normalization can come back `unknown`. |
| LayerNorm | **Race checks work; equivalence unclear** | `sqrt`/`rsqrt` get lowered (lowering.rs), but the paper's equivalence benchmarks contain no LayerNorm. Needs an experiment. |

## Key limits that matter for us
- Real-number semantics: blind to FP overflow/rounding → this is the gap change-aware tests cover.
- Proves only CTA-0 under fixed array sizes → boundary bugs in *other* CTAs / non-divisible shapes
  can be missed unless we pick shapes so that CTA-0 sees the boundary. Change-aware tests cover this too.
- No async-copy pipelines → "sync / pipeline stage" bugs with `cp.async` are out of VOLTA's scope.

## Next steps
1. Write tiny CUDA versions of softmax and LayerNorm, compile to PTX with `nvcc -arch=sm_80 -ptx`, and run `volta compare` on each.
2. Define "verification coverage" = fraction of candidates where VOLTA returns EQUIV/NOT-EQUIV/RACE (i.e. not unsupported/unknown/timeout).
