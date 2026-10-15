# STATUS

Last update: 2026-10-15

| Task | Status |
|---|---|
| RL00 | READY |
| RL01 | DONE |
| RL02 | DONE |
| RL03 | DONE |
| RL04 | **PASS**（附 PPO 预算 caveat） |
| RL05–RL14 | TODO |

## RL04 gate 说明（2026-10-15）

- 环境：`Pendulum-v1`，Gymnasium 1.4.0 + Stable-Baselines3 2.9.0 + PyTorch 2.14.1(cpu)，Python 3.13.12，独立 venv。
- 第一轮 prescribed 实验（seed=42, 50k steps）已完成：
  - **SAC**：独立评估 mean return **-151.2**，upright fraction **84.0%** → 稳定学会摆杆起摆控制。
  - **PPO**：独立评估 mean return **-1101.1**，upright fraction **0.75%** → 50k 步内**未学会**，接近 random 基线（-1258.8）。
- 诊断 + 修复：定位根因为"PPO 训练预算不足"（非环境/评估/代码缺陷，SAC 在同环境同协议下成功）。
  扩展 PPO 至 **300k** 步（超参数不变）后评估 mean return **-227.3**，upright fraction **75.0%** → 学会。
- 多随机种子（42/43/44）复核：
  - PPO@50k：-1152.9 ± 57.2，upright 0.62%（3/3 种子均失败）
  - SAC@50k：-150.6 ± 0.8，upright 83.86%（3/3 种子均成功）
- 产物：`RL04/results/rl04/`、`RL04/scripts/`、`RL04/configs/`、`RL04_REPORT.md`。

Current gate: **可以进入 RL05（FR3 Reach）**；建议在 RL05 中启用 observation 归一化并对 PPO 使用更大预算。
