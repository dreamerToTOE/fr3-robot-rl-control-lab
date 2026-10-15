#!/usr/bin/env bash
# RL04 Pendulum 连续控制实验 —— 一键复现脚本
#
# 用法:
#   bash run_all.sh                # 完整流程: smoke -> train -> eval -> sweep -> plot -> video
#   bash run_all.sh train          # 只训练
#   bash run_all.sh eval           # 只做最终评估 + 对照
#   bash run_all.sh plot           # 只出图
#
# 事先准备虚拟环境（已在本机创建）:
#   /home/ubuntu2004/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
#       -m venv /home/ubuntu2004/.workbuddy/binaries/python/envs/rl04
#   export TMPDIR=/home/ubuntu2004/.workbuddy/tmp      # 本机 /tmp 仅 10MB tmpfs
#   pip install --index-url https://download.pytorch.org/whl/cpu torch
#   pip install gymnasium stable-baselines3 numpy matplotlib pandas pyyaml imageio imageio-ffmpeg
set -euo pipefail

PY="${PY:-/home/ubuntu2004/.workbuddy/binaries/python/envs/rl04/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

STEP="${1:-all}"

smoke() {
  echo "=== [1/6] smoke test ==="
  "$PY" scripts/smoke_test.py
}

train() {
  echo "=== [2/6] train PPO seed=42 (50000 steps) ==="
  "$PY" scripts/train.py --algo ppo --config configs/ppo_seed42.yaml
  echo "=== [3/6] train SAC seed=42 (50000 steps) ==="
  "$PY" scripts/train.py --algo sac --config configs/sac_seed42.yaml
}

evaluate() {
  echo "=== [4/6] independent evaluation (20 episodes each) ==="
  "$PY" scripts/eval.py --algo random --episodes 20 --seed-base 1000
  "$PY" scripts/eval.py --algo ppo    --episodes 20 --seed-base 1000
  "$PY" scripts/eval.py --algo sac    --episodes 20 --seed-base 1000
  echo "=== [5/6] checkpoint sweep (eval return vs steps) ==="
  "$PY" scripts/eval.py --algo ppo --mode sweep --episodes 20 --seed-base 1000
  "$PY" scripts/eval.py --algo sac --mode sweep --episodes 20 --seed-base 1000
}

plot_all() {
  echo "=== [6/6] plots + comparison ==="
  "$PY" scripts/plot.py
}

video() {
  "$PY" scripts/record_video.py --algo ppo --seed 1000 || true
  "$PY" scripts/record_video.py --algo sac --seed 1000 || true
}

case "$STEP" in
  smoke) smoke ;;
  train) train ;;
  eval)  evaluate; plot_all ;;
  plot)  plot_all ;;
  all)   smoke; train; evaluate; plot_all; video ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
echo "=== done: $STEP ==="
