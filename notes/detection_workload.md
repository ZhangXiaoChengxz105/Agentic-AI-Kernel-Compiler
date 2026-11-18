# Detection workload: the 11 injected bugs and the 34 hard edits (2026-11-18)

This note lists every wrong kernel in the detection workload of the progress report (`notes/progress_report.md`,
Section 3) and says what is wrong with it. All numbers come from the final A100 runs:
`results/eval/bugset_tirx_gpu_20261117-053113.json` (injected bugs) and
`results/eval/mutants_detect_tirx_{bases,heldout,fresh}.json` (generated edits). Nothing was re-run; the 34 are a
subset of the 164 edits in those files.

A = compiler checks + fixed tests, B = A + VOLTA at one default size, C = A + change-aware tests + VOLTA at the
change-aware sizes. "Fixed tests" use power-of-two sizes: row ops rows 64 × N ∈ {256, 1024, 4096}, GEMM 128³, 256³,
512³, normal inputs.

## Why fixed tests miss these kernels
Almost every kernel below fails in one of three ways that fixed tests cannot see:
1. **Only at sizes that are not a multiple of the tile.** The fixed sizes are all multiples of 128 (row ops) or 16
   (GEMM), so a wrong loop bound or guard never triggers.
2. **A memory error or race whose output stays right.** Reading past an array, reading uninitialized shared memory,
   or many threads writing the same value to one address. The output is correct, but the kernel is wrong by CUDA's
   rules and may break on another GPU or compiler version.
3. **Float-only failures.** Softmax gives the same answer whatever number is subtracted before `exp`, so over real
   numbers a wrong max is harmless. In float32, `exp` then overflows or underflows on large inputs and the output
   becomes inf / NaN. Normal inputs never trigger it.

## The 11 injected bugs (hand-written)
Each kernel has a tail bug, an off-by-one and a missing barrier; softmax and LayerNorm each have one more numerical
bug. Sources: `workloads/<op>/tirx/bugs/`.

| bug | what is wrong | A | B | C |
|---|---|---|---|---|
| `softmax_txl_bug_tail` | the max and sum loops cover only full 128-wide tiles, so the last N mod 128 elements are dropped (type 1) | ✗ | ✗ | ✓ |
| `softmax_txl_bug_offbyone` | the sum loop guard is `col <= n`: it adds one element past the row (type 1, caught by memcheck) | ✓ | ✓ | ✓ |
| `softmax_txl_bug_nosync` | the barrier after each tree-reduction step is removed: race, wrong output | ✓ | ✓ | ✓ |
| `softmax_txl_bug_nomax` | `exp(x)` without subtracting the row max: overflows for large x (type 3) | ✗ | ✗ | ✓ |
| `layernorm_txl_bug_tail` | both reduction loops visit only full 128-wide tiles, dropping the last N mod 128 elements from mean and variance (type 1) | ✗ | ✗ | ✓ |
| `layernorm_txl_bug_offbyone` | the variance loop covers col = 0..n, adding the next row's first element: wrong at every size | ✓ | ✓ | ✓ |
| `layernorm_txl_bug_onepass` | variance as E[x²] − mean²: equal over the reals, but catastrophic cancellation in float32 when \|mean\| ≫ std (type 3) | ✗ | ✗ | ✓ |
| `layernorm_txl_bug_nosync` | the barrier between reading the mean from `red[0]` and reusing `red` is removed: race (type 2, caught by racecheck) | ✓ | ✓ | ✓ |
| `gemm_txl_bug_ktail` | the tile count is k // 16 instead of ceil(k / 16): the last k mod 16 products are dropped (type 1) | ✗ | ✗ | ✓ |
| `gemm_txl_bug_guard` | the output guard uses `<=`: threads write one row / column past C (type 1, caught by memcheck) | ✓ | ✓ | ✓ |
| `gemm_txl_bug_nosync` | the barrier after the inner-product loop is removed: fast threads overwrite the shared tiles too early, wrong output | ✓ | ✓ | ✓ |
| **caught** | | **6** | **6** | **11** |

Fixed tests alone catch 3 (the two output-changing nosync bugs and the LayerNorm off-by-one). The compiler checks add
3; VOLTA at its default size adds none (the tail bugs need non-multiple sizes, the numerical bugs are float-only);
the change-aware tests add the 3 tail bugs and the 2 numerical bugs.

## The 34 hard edits (generated)

**How they were selected.** A script changed one place in 9 correct kernels (the softmax, LayerNorm and GEMM
references and optimized versions, plus 4 softmax kernels written by GLM-5.3), giving 164 edits. We dropped:
- 119 that plain fixed tests already catch: 36 do not compile (26 of them change `float32` to `float16`) and 83 give
  visibly wrong output on normal inputs;
- 11 that cannot change any output (listed at the end).

The 34 left all change the kernel's behaviour, and fixed tests catch none of them.

**The archive.** The 130 dropped edits are not deleted: the full run stays in the result files above, and
`akc mutants --detect` skips them unless `--archived` is given. The list of the 34 (and of the 11 below) is
`akc/evaluation/hard_edits.py`; `python -m akc.evaluation.hard_edits` re-selects them from the full run and writes
`results/eval/hard_edits.json`.

| | A | B | C |
|---|---|---|---|
| caught (/34) | 19 | 29 | 33 |

### By kind of problem

**1. `<` → `<=` in a guard (15).**
- 13 guards on a column / row / k index (`col < n`, `base < n`, `idx < n`, `row < m`, `kb < k`). At tile-multiple
  sizes the index never reaches the bound, so nothing changes. At other sizes the kernel reads one element past the
  row (GEMM: writes one row or column past C). Type 1 + type 2. All 13 are caught by memcheck at the compiler checks'
  non-multiple sizes (A).
- 2 tree-reduction guards (`tid < s`, `tid < k`). One extra thread also combines `red[2s]`; in the first step that
  is one past the 128-slot shared array, and it races with the thread that owns that slot. Type 2, caught by
  racecheck / initcheck (A).

**2. Halved sizes (4).**
- `BLOCK = 128` → `64` (2): the block now has 64 threads and a 64-slot shared array, but the tree reduction still
  starts at stride 64, so it reads 64 slots past the array (uninitialized). Type 2, caught by initcheck (A).
- A shared array halved (2: LayerNorm `red`, GEMM `Bs`): accesses go past the shared allocation, but the output
  happens to stay right. Type 2. Only VOLTA's out-of-bounds analysis sees it (B).

**3. Races (9).**
- A deleted barrier (2): a thread can overwrite `red` before the others have read it. Type 2, caught by racecheck (A).
- A negated single-writer guard (7: `if (lane == 0)` six times in the GLM kernels, `if (tid == 0)` once in the
  LayerNorm reference): every other thread then writes the same value to the same address. That is a race by the
  memory model, but the output never changes and racecheck does not report it. Only VOLTA sees it (B).

**4. Numerical (5).**
- Float-only (4, type 3), caught only by the change-aware tests' extreme and outlier inputs (C):
  - softmax max → min;
  - a max initialized to 0 instead of −FLT_MAX (2): on rows of large negative values every `exp` underflows to 0,
    and 0 / 0 gives NaN;
  - a warp-shuffle max that starts at offset 8 instead of 16, so it takes the max over half the lanes.
- LayerNorm `+ EPS` → `− EPS` (1): a real value change, but about 1e-5, inside the tolerance for normal inputs. On a
  constant row the variance is 0 and `rsqrt(−EPS)` is NaN. VOLTA proves it not equivalent (B); the change-aware tests
  also catch it.

**5. Out of bounds only at N = 1 (1), missed by every configuration.** In the LayerNorm reference, `x[r, 0]` →
`x[r, 0 + 1]`. That element is only a shift for the mean and cancels out, so the output stays right. At N = 1 it reads
past the row (past the tensor on the last row). No configuration tests N = 1: the compiler checks use N = T+1 and 1000.

### All 34

Kernels: `softmax_txl_ref` (tuning set), `glm_txl_*` (4 GLM softmax kernels, held-out set), `layernorm_txl_*` and
`gemm_txl_*` (fresh set). "First caught by" is the first configuration in A → B → C that catches the edit.

| # | kernel:line | edit | kind | first caught by | evidence (in C unless noted) |
|---|---|---|---|---|---|
| 1 | softmax_txl_ref:30 | `tid < s` → `<=` | 1 reduction guard | A | racecheck, initcheck; VOLTA out of bounds |
| 2 | softmax_txl_ref:43 | `col < n` → `<=` | 1 guard | A | memcheck; change-aware inf/NaN |
| 3 | softmax_txl_ref:51 | `col < n` → `<=` | 1 guard | A | memcheck; change-aware mismatch; VOLTA not equivalent |
| 4 | softmax_txl_ref:8 | `BLOCK = 128` → `64` | 2 halved block | A | initcheck; change-aware inf/NaN; VOLTA out of bounds |
| 5 | softmax_txl_ref:37 | barrier deleted | 3 race | A | racecheck; change-aware mismatch; VOLTA race |
| 6 | softmax_txl_ref:46 | max → min | 4 float-only | C | change-aware inf/NaN |
| 7 | glm_txl_C_r2:95 | `col < n` → `<=` | 1 guard | A | memcheck; change-aware inf/NaN; VOLTA out of bounds |
| 8 | glm_txl_C_r2:4 | max init −FLT_MAX → 0 | 4 float-only | C | change-aware inf/NaN |
| 9 | glm_txl_C_r2:70 | `if (lane == 0)` negated | 3 same-value race | B | VOLTA race |
| 10 | glm_txl_D1_r2:148 | `base < n` → `<=` | 1 guard | A | memcheck; change-aware inf/NaN; VOLTA out of bounds |
| 11 | glm_txl_D1_r2:63 | `if (lane == 0)` negated | 3 same-value race | B | VOLTA race |
| 12 | glm_txl_D2_r1:151 | `base < n` → `<=` | 1 guard | A | memcheck, racecheck; change-aware inf/NaN; VOLTA race |
| 13 | glm_txl_D2_r1:127 | `if (lane == 0)` negated | 3 same-value race | B | VOLTA race |
| 14 | glm_txl_D2_r1:138 | `if (lane == 0)` negated | 3 same-value race | B | VOLTA race |
| 15 | glm_txl_D3_r1:97 | `idx < n` → `<=` | 1 guard | A | memcheck, racecheck; change-aware inf/NaN; VOLTA race |
| 16 | glm_txl_D3_r1:65 | warp max offsets start at 8, not 16 | 4 float-only | C | change-aware inf/NaN |
| 17 | glm_txl_D3_r1:12 | max init −FLT_MAX → 0 | 4 float-only | C | change-aware inf/NaN |
| 18 | glm_txl_D3_r1:109 | `if (lane == 0)` negated | 3 same-value race | B | VOLTA race |
| 19 | glm_txl_D3_r1:141 | `if (lane == 0)` negated | 3 same-value race | B | VOLTA race |
| 20 | layernorm_txl_opt:36 | `tid < k` → `<=` | 1 reduction guard | A | racecheck, initcheck; VOLTA out of bounds |
| 21 | layernorm_txl_opt:50 | `col < n` → `<=` | 1 guard | A | memcheck; change-aware mismatch; VOLTA not equivalent |
| 22 | layernorm_txl_opt:60 | `col < n` → `<=` | 1 guard | A | memcheck; change-aware mismatch; VOLTA not equivalent |
| 23 | layernorm_txl_opt:11 | `BLOCK = 128` → `64` | 2 halved block | A | initcheck; change-aware inf/NaN; VOLTA out of bounds |
| 24 | layernorm_txl_opt:43 | barrier deleted | 3 race | A | racecheck; change-aware mismatch; VOLTA race |
| 25 | layernorm_txl_opt:64 | `+ EPS` → `− EPS` | 4 small value change | B | VOLTA not equivalent; change-aware inf/NaN |
| 26 | layernorm_txl_opt:25 | shared `red` halved | 2 halved shared array | B | VOLTA out of bounds |
| 27 | gemm_txl_ref:27 | `row < m` → `<=` | 1 guard | A | memcheck; change-aware out of bounds; VOLTA out of bounds |
| 28 | gemm_txl_ref:28 | `col < n` → `<=` | 1 guard | A | memcheck; change-aware out of bounds; VOLTA out of bounds |
| 29 | layernorm_txl_ref:36 | `x[r, 0]` → `x[r, 0 + 1]` | 5 out of bounds at N = 1 | – | none |
| 30 | layernorm_txl_ref:35 | `if (tid == 0)` negated | 3 same-value race | B | VOLTA race |
| 31 | gemm_txl_opt:39 | `row < m` → `<=` | 1 guard | A | memcheck; VOLTA out of bounds |
| 32 | gemm_txl_opt:44 | `kb < k` → `<=` | 1 guard | A | memcheck; change-aware inf/NaN; VOLTA out of bounds |
| 33 | gemm_txl_opt:54 | `row < m` → `<=` | 1 guard | A | memcheck; change-aware out of bounds; VOLTA out of bounds |
| 34 | gemm_txl_opt:31 | shared `Bs` halved | 2 halved shared array | B | VOLTA out of bounds |

### The 11 dropped edits that cannot change any output
- 9 change a block size, tile size or unroll factor in a kernel that derives everything else from it (GLM C_r2 lines
  9 ×2 and 11, GLM D2_r1 line 19 ×2, GEMM ref line 10 ×2, GEMM opt line 11 ×2). VOLTA proves the 5 GLM ones
  equivalent; it cannot decide the 4 GEMM ones (block 0 then covers another part of the output), which we judged
  harmless by reading the code.
- 2 pad the GEMM shared tiles with 1.0 instead of 0.0 (GEMM opt lines 38, 43). The padding is always multiplied by
  the other operand's zero padding. VOLTA proves both equivalent.

No configuration flags any of them, as expected, and none flags any of the 27 renames.
