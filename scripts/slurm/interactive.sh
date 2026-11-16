#!/bin/bash
# Interactive shell on one A100 (gpuA100x4-interactive, max 1 h). ASK BEFORE RUNNING (uses the allocation).
#   bash scripts/slurm/interactive.sh            # default 30 min
#   TIME=00:15:00 bash scripts/slurm/interactive.sh
# Inside the session:  cd <repo> && source env.sh      (activates envs/akc)
set -euo pipefail
TIME="${TIME:-00:30:00}"
cmd=(srun --account=bidr-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --ntasks=1
     --gpus-per-node=1 --cpus-per-task=16 --mem=32g --time="$TIME" --pty bash)
echo "about to run: ${cmd[*]}"
read -r -p "Start the interactive GPU session? [y/N] " ok
[[ "$ok" == [yY] ]] && exec "${cmd[@]}"
echo "cancelled"
