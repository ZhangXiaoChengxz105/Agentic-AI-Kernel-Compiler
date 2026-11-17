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
| Tiled GEMM | **Supported (verified 2026-11-15)** | Our naive ref vs shared-memory tiled opt (TILE=16) EQUIV at (32,32,48), (32,32,40), (12,12,40) in ≤0.15 s. Edge-of-matrix bugs are only visible if CTA 0 touches the edge (M or N < 16). Must avoid `cp.async`/TMA pipelines. |
| Row-wise softmax | **Supported (verified 2026-11-15)** | Our ref (1 thread/row) vs opt (128-thread tree reduction, `*1/sum`) proved EQUIV at N=200/256/1000/1024 in ≤0.5 s. Requires `exp` as `__symexpf` and PTX without `-lineinfo`. See `notes/volta_loop_log.md` |
| LayerNorm | **Supported (verified 2026-11-15)** | Two-pass ref vs block-parallel opt (rsqrt) EQUIV at N=128 (1.8 s) and N=100 (0.6 s). Cost grows fast with N (variance expands to O(N²) terms): at N≥512 a run took >3.5 min and ~10 GB RAM (stopped). Keep VOLTA shapes ≤ 200. One-pass variance (E[x²]−mean²) is proved EQUIV: blind spot. |

## Key limits that matter for us
- Real-number semantics: blind to FP overflow/rounding → this is the gap change-aware tests cover.
- Proves only CTA-0 under fixed array sizes → boundary bugs in *other* CTAs / non-divisible shapes
  can be missed unless we pick shapes so that CTA-0 sees the boundary. Change-aware tests cover this too.
- No async-copy pipelines → "sync / pipeline stage" bugs with `cp.async` are out of VOLTA's scope.

## Next steps
1. Write tiny CUDA versions of softmax and LayerNorm, compile to PTX with `nvcc -arch=sm_80 -ptx`, and run `volta compare` on each.
2. Define "verification coverage" = fraction of candidates where VOLTA returns EQUIV/NOT-EQUIV/RACE (i.e. not unsupported/unknown/timeout).

## tirx-lite kernels (Task 3, 2026-11-17)

**Result.** VOLTA decides all 23 tirx-lite softmax kernels we tried, at N = 256, 584 and 200: 69 of 69 runs give a
verdict (coverage 100%). This needs five rewrites of TIRx's generated code (table below).
- Script: `scripts/experiments/volta_tirx.py`. Results: `results/eval/volta_tirx.json` (Slurm job 24060621, CPU work
  on an A100 node, 3 min).
- Ablation at N = 256 on the reference + 4 GLM kernels, each rewrite switched off in turn:
  `results/eval/volta_tirx_ablation.json` (job 24060724, 1.5 min).
  - With all rewrites: 5/5 decided.
  - Without one rewrite: exp 0/5, ex2 1/5, huge_const 0/5, bar_count 1/5, select_max 4/5.
  - So every rewrite is needed by at least one kernel.

**How a tirx-lite candidate is checked** (`akc/validation/volta/tirx.py`, `VoltaValidator` for `lang="tirx"`):
- Trace the candidate and the tirx reference `softmax_txl_ref` at the VOLTA size, with rows = 2. Shapes are static,
  so each size is a separate trace.
- Take the generated CUDA (`Kernel.source()`, CPU only, about 2 s per kernel and size).
- Rewrite it (below), then `nvcc -arch=sm_80 -ptx -DVOLTA`, without `-lineinfo` (same as the CUDA path).
- Launch settings come from the kernel itself: grid and block from `kernel.session.launch`, dynamic shared bytes
  from the traced `cuda.smem_required`.
- Kernel parameters are the tensor pointers only (`x_ptr, y_ptr`). They are mapped to the manifest's
  `tirx.tensors` by their position among the traced function's parameters.
- `volta compare` candidate vs reference. VOLTA takes one dynamic-shared size for both kernels. When the candidate
  needs less than the reference, it is first checked alone (`volta analyze`) at its own size.
- Verdicts become findings exactly as on the CUDA path (same statuses, hints, mismatching elements).

**What VOLTA cannot read in TIRx output, and the rewrite** (applied the same way to candidate and reference;
the counts are kept in each run's `rewrites`):

| construct (where it comes from) | VOLTA without the rewrite (ablation) | rewrite | exact? |
|---|---|---|---|
| `expf(x)` (`txl.exp`) | nvcc expands it to `ex2.approx.ftz` + range reduction: "lowering failed: Unsupported instruction: Ex2" | `__symexpf(x)` | yes, in VOLTA's real-number model (same as `EXP()` in our CUDA) |
| inline `ex2.approx.f32` (all 4 GLM kernels: `ex2((x - m) * LOG2E)`) | "lowering failed: Unsupported instruction: Ex2" | `__symexpf(t / fl32(log2 e))` | no: ln 2 is taken as 1/fl32(log2 e), the exponent is off by 1.3e-8 relative. A missing or wrong LOG2E factor is still caught (unit test) |
| float constants with magnitude >= 2^127: `-3.4e38` (reference), `-FLT_MAX` (GLM), also as uint bits reinterpreted as float | "rational coefficient overflowed i128" (every pair, since the reference has one) | +-inf | only when the constant is the identity of a max/min and every input lies above it (true in all our kernels) |
| `bar.sync %r, %r` (`txl.ptx.bar.sync(0, 128)` passes registers; all 4 GLM kernels) | "lowering failed: Unsupported instruction: bar.sync" (reason: register barrier id) | `bar.sync 0` when the count equals the CTA size | yes (PTX: without a count all CTA threads take part) |
| compare + select (`txl.Select(a > b, a, b)`, and `txl.max`, which is a NaN-propagating select) | opaque term; with a -inf identity (GLM C): "non-finite constant -inf in expression" | `max.f32` / `min.f32` in the PTX | yes over the reals; only NaN handling differs in float32 |

**Results** (statuses at N = 256 / 584 / 200):

| kernel | VOLTA | CUDA counterpart |
|---|---|---|
| softmax_txl_ref vs itself | EQUIV / EQUIV / EQUIV | - |
| softmax_txl_bug_nosync | RACE x3 | softmax_bug_nosync: RACE |
| softmax_txl_bug_tail | EQUIV / NOT_EQUIV / NOT_EQUIV | same: hidden when N % 128 == 0 |
| softmax_txl_bug_offbyone | EQUIV / NOT_EQUIV / NOT_EQUIV | NOT_EQUIV at every N |
| softmax_txl_bug_nomax | EQUIV x3 | EQUIV (real-number blind spot) |
| glm_txl_C_r2, glm_txl_D1_r2, glm_txl_D2_r1, glm_txl_D3_r1 | EQUIV x3 each | - |

- 26 of 27 checks match the expectation. The one difference is the tirx off-by-one at N = 256. The tirx loop runs a
  fixed `ceil(n / 128)` tiles, so `col <= n` can reach `col == n` only when n % 128 != 0. The CUDA loop
  `for (i = tid; i <= N; i += 128)` reaches i = N at every N. So the tirx bug does not exist at N = 256; it is not a
  VOLTA miss.

**The 14 tirx mutants that the tests (A and C) missed** (`mutants_detect_{bases,heldout}.json`):
- VOLTA catches 8, at all three sizes:
  - all 5 races named in the README:
    - `if (lane == 0)` negated (C L70, C L81, D1 L63, D2 L127): RACE. Every lane writes the same value to the same
      shared address;
    - `tid < s` -> `tid <= s` (softmax_txl_ref L30): OUT_OF_BOUNDS. Thread 64 reads `red[128]` at s = 64;
  - `base < n` -> `base <= n` in GLM D2 L151 and `idx < n` -> `idx <= n` in D3 L97: RACE. These were filed as
    "reads one element past the row". They also store that vector past the staged row in shared memory, into the
    reduction scratch, with no barrier before the warp leaders write there;
  - `BLOCK = 128` -> `64` in softmax_txl_ref (L8): OUT_OF_BOUNDS. This was filed as "equivalent". But the tree
    reduction hard-codes its first step s = 64, so it reads `red[64..127]` of a 64-float array.
- VOLTA proves 6 equivalent:
  - 4 block-size edits in GLM C and D2, which derive everything from the block size;
  - the float16 accumulator (C L40): a real-number blind spot;
  - `[r, col]` -> `[r, col + 1]` (C L114): it is in the n > 12284 code path, which VOLTA sizes never run.

**Cost.** VOLTA itself takes 0.01 to 3.3 s per run, 9.3 s for all 69 runs. Tracing, code generation and nvcc take
about 2 s per kernel and size (144 s in total) and dominate.

**Limits** (in addition to the CUDA ones: real numbers, CTA 0 only, exact sizes):
- The `ex2` and `huge_const` rewrites are modelling assumptions, as stated above.
- Only code paths reached at the VOLTA sizes are checked; GLM's large-row paths (n > 12284) are not.
- Still unsupported, with a hint to the agent: data-dependent control flow, partial-CTA barriers, `cp.async` /
  `mbarrier`, transcendental functions other than exp and ex2, and kernels with scalar parameters.
- (Update 2026-11-17: LayerNorm and GEMM have tirx-lite kernels too; see the next section.)

## tirx-lite LayerNorm and GEMM (Task 1, 2026-11-17)

**Result.** VOLTA decides every tirx-lite LayerNorm and GEMM kernel at every VOLTA shape, with the verdict the
manifest expects. No code change in the VOLTA path was needed: `make_tirx_case` already takes the arrays from the
manifest's `tirx.tensors`, and each kernel's own grid (GEMM: 2-D).
- Script: `scripts/experiments/volta_tirx.py --workload {layernorm,gemm} --groups reference,opt,bug`.
- Results: `results/eval/volta_tirx_{layernorm,gemm}.json` (Slurm job 24061108, rerun of 24061076).
- Shapes: the manifest default and deep shapes, plus the change-aware tiling and sync shapes:
  - LayerNorm: N = 128, 200, 100 (rows = 2; max_N = 200);
  - GEMM: (M, N, K) = (32, 32, 48), (32, 32, 72), (32, 32, 40), (12, 12, 40), (16, 16, 64).
- None of the five rewrites fire: these kernels use no exp, ex2, huge constants, counted barriers or selects.

| workload | runs | decided | as expected | VOLTA s | build s |
|---|---|---|---|---|---|
| LayerNorm (6 kernels x 3 shapes) | 18 | 18 | 18/18 | 30.9 | 36.2 |
| GEMM (5 kernels x 5 shapes) | 25 | 25 | 25/25 | 2.2 | 41.5 |

**Verdicts per kernel** (same as the CUDA counterparts):

| kernel | VOLTA |
|---|---|
| layernorm_txl_ref vs itself, layernorm_txl_opt | EQUIV at N = 128, 200, 100 |
| layernorm_txl_bug_nosync | RACE x3 (on red[0]) |
| layernorm_txl_bug_tail | EQUIV at N = 128 (hidden), NOT_EQUIV at 200 and 100 |
| layernorm_txl_bug_offbyone | NOT_EQUIV x3 (the port covers col = 0..n, like CUDA's i <= N, so it shows at every N) |
| layernorm_txl_bug_onepass | EQUIV x3 (real-number blind spot) |
| gemm_txl_ref vs itself, gemm_txl_opt | EQUIV x5 |
| gemm_txl_bug_nosync | RACE x5 (every shape has K > 16) |
| gemm_txl_bug_ktail | NOT_EQUIV exactly when K % 16 != 0 (K = 72, 40, 40); EQUIV at K = 48, 64 |
| gemm_txl_bug_guard | RACE only at M = N = 12 (CTA 0 touches the edge); EQUIV otherwise |

**Notes.**
- The LayerNorm tirx reference is serial: thread 0 of a 32-thread CTA does the whole row. It computes the mean as
  x0 + sum(x - x0) / n. Over the reals this equals sum(x) / n, and VOLTA proves it.
- LayerNorm VOLTA time grows with N as expected: about 0.5 s at N = 100, 1.2 s at 128 and 4-5 s at 200.

**Softmax tirx bugs at the new change-aware shapes** (`results/eval/volta_tirx_softmax_task1.json`, N = 256, 584,
200, 100, 264): 24 of 25 runs as expected (the manifest now has `expect_volta` for the 4 tirx bugs).
- The exception is softmax_txl_bug_tail at N = 100: UNSUPPORTED ("non-finite constant -inf in expression").
  - With n < 128 the buggy loops run n // 128 = 0 tiles, so the row max keeps its identity -3.4e38.
  - The huge_const rewrite turns that into -inf, and exp(x - (-inf)) cannot be folded.
  - So VOLTA is undecided here, not wrong.
  - The CUDA twin is caught at such sizes, as DIV_BY_ZERO / NOT_EQUIV.
- Coverage for the tirx kernels at these shapes: 67 of 68 runs decided (LayerNorm 18, GEMM 25, softmax 24 of 25).
