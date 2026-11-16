# GLM-5.3 pilot: can the agent model write a correct softmax? (2026-11-15)

Script: `scripts/pilot_glm_softmax.py` (mini repair loop: prompt → GLM → extract code → CPU evaluation →
feedback, ≤ 3 rounds). Raw replies, code and evaluations per round: `results/pilot/<timestamp>/`;
every Lumen call: `logs/lumen/calls.jsonl`. Model `glm-5.3` via Lumen, `reasoning_effort=high`, `max_tokens=100000`.

Evaluation (no GPU yet):
- **CUDA**: nvcc sm_80 → VOLTA vs `kernels/softmax/softmax_ref.cu` at N = 256, 200, 100 (equivalence + races).
- **TIRx**: safety scan → trace → sm_80 compile → NumSim (CPU execution) vs float64 softmax with N(0,1) and
  ×100 inputs at n = 200, 256 → Racecheck (`agent/tirx_eval.py`; validated on a hand-written TIRx softmax and
  on 4 injected TIRx bugs, all detected).

## Results
| run | lang | rounds | outcome per round | coins | 1st-reply latency |
|---|---|---|---|---|---|
| A0 | CUDA | 1 (stopped) | default effort: 24k output tokens of reasoning, no code (`finish=length`) | 0.102 | 495 s |
| A | CUDA | 3 | r1, r2: VOLTA `i128 overflow` (kernel was correct: −FLT_MAX constant, see below) → r3 EQUIV | 0.088 | 71 s |
| B1 | CUDA | 3 | r1: **real data race** (VOLTA: t0 write vs t32 read, Shared[0x0]) → r2: race fixed, −FLT_MAX overflow → r3 EQUIV (used the new hint) | 0.041 | 19 s |
| B2 | CUDA | 1 | r1 EQUIV | 0.016 | 22 s |
| B3 | CUDA | 1 | r1 EQUIV | 0.009 | 12 s |
| C | TIRx | 2 | r1 trace error (`else_` not at top of `if_`) → r2 ok | 0.139 | 107 s |
| D1 | TIRx | 2 | r1 same trace error → r2 ok | 0.172 | 726 s |
| D2 | TIRx | 1 | r1 ok (max err 9e-9, x100 inputs 3.5e-8, racecheck clean) | 0.092 | 90 s |
| D3 | TIRx | 1 | r1 ok | 0.067 | 57 s |
Total: 15 requests, 0.727 coins (balance 9.27).

| (high effort runs only) | CUDA (4 runs) | TIRx (4 runs) |
|---|---|---|
| passed within 3 rounds | 4/4 | 4/4 |
| passed in round 1 (as evaluated) | 2/4 | 2/4 |
| round-1 code functionally correct | 3/4 (A's r1 was correct; VOLTA false alarm) | 2/4 |
| round-1 failure kinds | verifier limitation ×1, real race ×1 | TIRx API rule (`else_` placement) ×2 |
| mean coins per run | 0.039 | 0.118 |
| round-1 latency | 12–71 s | 57–726 s |

## Findings
1. **GLM-5.3 can write correct, optimized softmax kernels in both languages** (warp-shuffle reductions,
   shared-memory partials; the TIRx kernels cached the row in shared memory and used ex2.approx·log2e).
2. **Default reasoning effort is longer than `high`**: without the parameter, 24k tokens were spent thinking
   (8 min) with no code; with `high`, 2k–24k tokens and 12–726 s.
3. **The repair loop works**: one CUDA first draft had a genuine shared-memory race; VOLTA caught it and GLM
   fixed it from the feedback, which is exactly the project's thesis. GLM also diagnosed the VOLTA i128
   overflow from the raw error text alone (comment in its code), and after we added an explicit hint it fixed
   the same issue in one round.
4. **VOLTA limitation found**: constants with |c| > 2^127 ≈ 1.7e38 (e.g. −FLT_MAX = −3.4e38 used as max
   identity, very common in kernels) overflow VOLTA's exact i128 rationals → now reported as `UNSUPPORTED`
   with an actionable hint (`verify/hints.py`). Candidate upstream issue.
5. **TIRx first-draft errors are systematic** (both failures: `T.else_ must be called at the top of a T.if_()`),
   so a short "pitfalls" section in the TIRx prompt should remove most of them.
6. CUDA is ~3× cheaper and usually faster per run; CUDA output is directly VOLTA-verifiable, TIRx output
   currently only via NumSim + Racecheck (TIRx emits ex2.approx, which VOLTA does not treat as exp).

## Not yet checked
- Real GPU execution and speed of the GLM kernels (needs the GPU smoke job).
- Larger sample sizes; GEMM / LayerNorm; optimization (speed) objective rather than correctness only.
