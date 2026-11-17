#!/bin/bash
# Final A0/A/B/C evaluation on tirx-lite (Task 5): injected bugs and the three mutant sets.
# One part per job (the interactive A100 partition allows 1 h; one job at a time for the user, see CLAUDE.md):
#   sbatch --time=00:50:00 --mem=60g --job-name=tirx_eval scripts/slurm/gpu_job.sbatch bash scripts/slurm/tirx_eval.sh bugs_bases
#   ... tirx_eval.sh heldout      ... tirx_eval.sh fresh
# bugs_bases runs one mutant stream (its validator timings are the overhead numbers we report).
# heldout / fresh run two streams at once on the same GPU (two halves of the set, each with its own VOLTA pool) to
# halve the GPU time; their per-validator seconds are therefore a little inflated and are not used for overhead.
# VOLTA runs in parallel CPU workers (akc mutants --volta-workers); the GPU validators run serially per stream.
# Output: results/eval/bugset_tirx_gpu_<ts>.json, results/eval/mutants_detect_tirx_{bases,heldout,fresh}.json
set -uo pipefail
cd "${CS521:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUT=results/eval
LOG=logs/slurm
part="${1:?part: bugs_bases | heldout | fresh}"

stream() {   # stream <set> <bases,..> <out> <workers>
    akc mutants --set "$1" --lang tirx --only "$2" --detect --volta --volta-workers "$4" --out "$3" \
        > "$LOG/tirx_eval_${SLURM_JOB_ID:-local}_$(basename "$3" .json).log" 2>&1
}

merge() {    # merge <out> <part1> <part2>
    python - "$@" <<'EOF'
import json, sys
out, parts = sys.argv[1], sys.argv[2:]
ds = [json.load(open(p)) for p in parts]
m = {**ds[0], "rows": [r for d in ds for r in d["rows"]], "seconds": max(d["seconds"] for d in ds),
     "complete": all(d["complete"] for d in ds), "streams": len(ds), "parts": parts}
json.dump(m, open(out, "w"), indent=1)
print(f"merged {len(m['rows'])} rows from {len(parts)} streams -> {out} (complete={m['complete']})")
EOF
}

case "$part" in
  bugs_bases)
    echo "== akc eval-bugs --lang tirx"
    akc eval-bugs --lang tirx
    echo "== akc mutants --set bases (one stream)"
    akc mutants --set bases --lang tirx --detect --volta --volta-workers 12 --out $OUT/mutants_detect_tirx_bases.json
    ;;
  heldout)
    stream heldout glm_txl_C_r2,glm_txl_D1_r2 $OUT/mutants_detect_tirx_heldout_p1.json 6 &
    stream heldout glm_txl_D2_r1,glm_txl_D3_r1 $OUT/mutants_detect_tirx_heldout_p2.json 6 &
    wait
    tail -n 25 $LOG/tirx_eval_${SLURM_JOB_ID:-local}_mutants_detect_tirx_heldout_p*.log
    merge $OUT/mutants_detect_tirx_heldout.json $OUT/mutants_detect_tirx_heldout_p1.json $OUT/mutants_detect_tirx_heldout_p2.json
    ;;
  fresh)
    stream fresh layernorm_txl_opt,gemm_txl_ref $OUT/mutants_detect_tirx_fresh_p1.json 6 &
    stream fresh layernorm_txl_ref,gemm_txl_opt $OUT/mutants_detect_tirx_fresh_p2.json 6 &
    wait
    tail -n 25 $LOG/tirx_eval_${SLURM_JOB_ID:-local}_mutants_detect_tirx_fresh_p*.log
    merge $OUT/mutants_detect_tirx_fresh.json $OUT/mutants_detect_tirx_fresh_p1.json $OUT/mutants_detect_tirx_fresh_p2.json
    ;;
  *) echo "unknown part $part"; exit 2 ;;
esac
