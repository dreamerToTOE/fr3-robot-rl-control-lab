# RL04 — Pendulum Continuous-Control Pipeline

## Goal
第一次完整跑通 train → checkpoint → eval → plot。

## Engineering
PPO 和 SAC 至少各跑一个小实验；统一 seed/config/result 结构。

## PASS
可从零命令复现实验，并输出 return/success-like metric/action 曲线。

---

# 实验记录（2026-10-15 执行）

## 1. 环境与 MDP 定义

| 项 | 值 |
|---|---|
| Environment | `Pendulum-v1` (Gymnasium 1.4.0) |
| Observation | `[cos(theta), sin(theta), theta_dot]`，shape `(3,)`，float32 |
| Action | 连续 torque `u ∈ [-2.0, 2.0]` N·m，shape `(1,)` |
| Reward | `-(theta^2 + 0.1*theta_dot^2 + 0.001*u^2)` |
| Termination | 无提前终止；固定 200 步后 TimeLimit **truncation** |
| Baseline | Random uniform policy（无需训练） |

## 2. 复现命令（从零）

```bash
# ① 环境（一次性）
/home/ubuntu2004/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
    -m venv /home/ubuntu2004/.workbuddy/binaries/python/envs/rl04
export TMPDIR=/home/ubuntu2004/.workbuddy/tmp          # 本机 /tmp 仅 10MB tmpfs
PY=/home/ubuntu2004/.workbuddy/binaries/python/envs/rl04/bin/python
$PY -m pip install --index-url https://download.pytorch.org/whl/cpu torch
$PY -m pip install gymnasium stable-baselines3 matplotlib pandas pyyaml imageio imageio-ffmpeg pygame-ce

# ② 一键全流程
cd RL04 && bash run_all.sh          # smoke -> train -> eval -> sweep -> plot -> video

# ③ 或分步
$PY scripts/smoke_test.py
$PY scripts/train.py --algo ppo --config configs/ppo_seed42.yaml
$PY scripts/train.py --algo sac --config configs/sac_seed42.yaml
$PY scripts/eval.py --algo random --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo ppo    --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo sac    --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo ppo --mode sweep --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo sac --mode sweep --episodes 20 --seed-base 1000
$PY scripts/plot.py
$PY scripts/record_video.py --algo sac --seed 1000

# ④ 附加实验
$PY scripts/train.py --algo ppo --config configs/ppo_seed42_extended.yaml --run-id ppo_seed42_300k
$PY scripts/train.py --algo ppo --config configs/ppo_seed42.yaml --seed 43 --run-id ppo_seed43
$PY scripts/train.py --algo sac --config configs/sac_seed42.yaml --seed 43 --run-id sac_seed43
$PY scripts/multiseed_summary.py
```

## 3. 结果路径

```text
RL04/results/rl04/
├── smoke_test.json
├── random_seed42/{eval_metrics.json,eval_episodes.csv,eval_trace.csv}
├── ppo_seed42/{config.json,metadata.json,final_model.zip,checkpoints/,train_metrics.csv,
│              eval_metrics.json,eval_episodes.csv,eval_trace.csv,eval_vs_steps.csv,plots/,logs/}
├── ppo_seed42_300k/          # 预算修复实验
├── ppo_seed43/  ppo_seed44/  sac_seed43/  sac_seed44/   # 多随机种子
└── comparison/
    ├── comparison.csv  comparison.md
    ├── multiseed_summary.csv  multiseed_summary.md
    └── plots/  (01..09)
```

## 4. 关键结果

| Policy | Mean Return | Upright frac | Mean \|theta\| (rad) | Saturation |
|---|---|---|---|---|
| Random | -1258.8 | 0.05% | 2.108 | 1.23% |
| PPO @ 50k（prescribed） | -1101.1 | 0.75% | 1.769 | 0.00% |
| PPO @ 300k（extended） | -227.3 | 75.00% | 0.470 | 7.38% |
| **SAC @ 50k（prescribed）** | **-151.2** | **84.00%** | **0.338** | 3.50% |

- 容差：upright = `|theta| ≤ 0.2 rad` 且 `|theta_dot| ≤ 1.0 rad/s`；saturation = `|u| ≥ 0.99×2.0 N·m`。
- 样本效率：SAC 在 **10k 步**即达 -158；PPO 到 **200k 步**仍为 -428。
- 训练耗时：PPO 50k ≈ 30s / SAC 50k ≈ 211s（CPU，16 核，单环境；SAC 每步做 1 次梯度更新且需重放采样）。

## 5. 结论

1. PPO 与 SAC **都完成了训练**（各 50k 步，seed=42；PPO 因 `n_steps=2048` 实际 51200 步）。
2. 与 Random 基线相比：**SAC 显著更优**；**PPO@50k 仅微弱更好**（≈ 失败）。
3. 累计 Reward 趋势：SAC 在前 10k 步快速上升后收敛至 -150 平台；PPO@50k 全程平坦，PPO@300k 呈持续上升。
4. 是否学会竖直控制：**SAC 学会**（84% upright）；**PPO@50k 未学会**（0.75%），**PPO@300k 学会**（75%）。
5. 力矩：SAC 均值 0.67 N·m、饱和率 3.5%，无长期饱和；PPO@50k 动作幅度极小（0.275 N·m）属"未学习导致的乏力"，PPO@300k 饱和率 7.4% 略高但不持续。
6. 样本效率：**SAC ≫ PPO**（本任务）；训练耗时 SAC 约为 PPO 的 **7 倍**（每步更新 vs 批式更新）。
7. 已由 3 个随机种子确认的结论：SAC@50k 稳定成功、PPO@50k 稳定失败。
   仍需更多种子/更大预算才能确认：PPO 成功所需的精确预算阈值与其最终性能上界。

## 6. 失败分析（PPO@50k）

- **不是 ENVIRONMENT**：smoke test 全通过，SAC 在同一环境同一 `make_env` 下成功。
- **不是 EVALUATION**：Random/PPO/SAC 使用完全相同的评估协议与种子，SAC 得到 84% upright。
- **不是 REWARD/EPISODE 处理**：reward 由 Gymnasium 原样给出，SAC 能收敛到接近最优 return。
- **根因 = RL 训练预算/更新次数不足**：50k 步 / `n_steps=2048` → 仅 25 次 PPO 更新，策略尚处早期。
- **一次修复**：超参数不变、预算扩至 300k → 学会（-227.3，upright 75%）。修复有效，未启用备选方案。

## 7. 已知限制

- 每种算法主实验为**单环境**训练（`n_envs=1`），PPO 因此每 2048 步才更新一次；多环境并行可提升 PPO 效率。
- 未使用 observation normalization（`VecNormalize`），这通常是 Pendulum 上 PPO 的关键增益项。
- 评估为 deterministic 20 episodes；未做 stochastic 策略评估对比。
- 训练曲线来自训练期 episode return（供观察），**不作为最终指标**；最终指标一律取独立评估。
