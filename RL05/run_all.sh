#!/usr/bin/env bash
# RL05 —— 一键复现流程（门禁 → 训练 → 评估 → 图表 → 视频）
#
# 用法:
#   bash run_all.sh              # 完整流程
#   bash run_all.sh gate         # 只跑门禁（benchmark/diagnose/targets/env tests）
#   bash run_all.sh train        # 只训练
#   bash run_all.sh eval         # 只评估 + 出图
set -uo pipefail

RL05=/home/ubuntu2004/lmy/fr3-robot-rl-control-lab/RL05
ISAAC=/home/ubuntu2004/isaacsim-4.5.0
export TMPDIR=/home/ubuntu2004/.workbuddy/tmp
export PYTHONPATH=$RL05/.pydeps
LOG=$RL05/results/rl05/pipeline.log
mkdir -p "$(dirname "$LOG")"

run() {
  echo "########## $* ##########" | tee -a "$LOG"
  ( cd "$ISAAC" && ./python.sh "$@" ) 2>&1 | tee -a "$LOG" | grep -aE "PASS|FAIL|_OK|_ERROR|success_rate|Targets|TARGETS|ENVTEST|DIAG|TRAIN|EVAL|VIDEO"
}

STEP="${1:-all}"
cd "$ISAAC"

case "$STEP" in
  gate)
    run "$RL05/tests/test_env_interface.py"
    ;;
  train)
    run "$RL05/scripts/train.py" --algo sac --config "$RL05/configs/sac_fr3_reach.yaml"
    run "$RL05/scripts/train.py" --algo ppo --config "$RL05/configs/ppo_fr3_reach.yaml"
    ;;
  eval)
    run "$RL05/scripts/eval_fr3_reach.py" --policy classical
    run "$RL05/scripts/eval_fr3_reach.py" --policy random
    run "$RL05/scripts/eval_fr3_reach.py" --policy sac --run-id sac_seed42
    run "$RL05/scripts/eval_fr3_reach.py" --policy ppo --run-id ppo_seed42
    python3 "$RL05/scripts/plot.py"
    ;;
  all)
    run "$RL05/tests/test_env_interface.py"
    run "$RL05/scripts/eval_fr3_reach.py" --policy classical
    run "$RL05/scripts/eval_fr3_reach.py" --policy random
    run "$RL05/scripts/train.py" --algo sac --config "$RL05/configs/sac_fr3_reach.yaml"
    run "$RL05/scripts/train.py" --algo ppo --config "$RL05/configs/ppo_fr3_reach.yaml"
    run "$RL05/scripts/eval_fr3_reach.py" --policy sac --run-id sac_seed42
    run "$RL05/scripts/eval_fr3_reach.py" --policy ppo --run-id ppo_seed42
    python3 "$RL05/scripts/plot.py"
    run "$RL05/scripts/record_video.py" --policy classical
    run "$RL05/scripts/record_video.py" --policy sac --run-id sac_seed42
    run "$RL05/scripts/record_video.py" --policy ppo --run-id ppo_seed42
    ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
echo "=== done: $STEP ===" | tee -a "$LOG"
