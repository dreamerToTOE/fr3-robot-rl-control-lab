#!/usr/bin/env bash
# RL05 —— 加大训练预算后的正式训练 + 评估（30k 试点诊断为预算不足后的"一次修复"）
set -uo pipefail

RL05=/home/ubuntu2004/lmy/fr3-robot-rl-control-lab/RL05
ISAAC=/home/ubuntu2004/isaacsim-4.5.0
PYPLOT=/home/ubuntu2004/.workbuddy/binaries/python/envs/rl04/bin/python
export TMPDIR=/home/ubuntu2004/.workbuddy/tmp
export PYTHONPATH=$RL05/.pydeps
LOG=$RL05/results/rl05/pipeline_train2.log
mkdir -p "$(dirname "$LOG")"

run() {
  echo "########## $* ##########" | tee -a "$LOG"
  ( cd "$ISAAC" && ./python.sh "$@" ) 2>&1 | tee -a "$LOG" \
    | grep -aE "TRAIN_OK|TRAIN_ERROR|EVAL_OK|EVAL_ERROR|Error:|error"
}

run "$RL05/scripts/train.py" --algo sac --config "$RL05/configs/sac_fr3_reach.yaml" \
    --run-id sac_seed42_r2 --total-timesteps 150000
run "$RL05/scripts/train.py" --algo ppo --config "$RL05/configs/ppo_fr3_reach.yaml" \
    --run-id ppo_seed42_r2 --total-timesteps 200000
run "$RL05/scripts/eval_fr3_reach.py" --policy sac --run-id sac_seed42_r2 --tag sac_eval
run "$RL05/scripts/eval_fr3_reach.py" --policy ppo --run-id ppo_seed42_r2 --tag ppo_eval
"$PYPLOT" "$RL05/scripts/plot.py" 2>&1 | tee -a "$LOG"
echo "=== done: train2 ===" | tee -a "$LOG"
