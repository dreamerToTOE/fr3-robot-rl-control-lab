# STATUS

Last update: 2026-10-16

| Task | Status |
|---|---|
| RL00 | READY |
| RL01 | DONE |
| RL02 | DONE |
| RL03 | DONE |
| RL04 | **PASS**（附 PPO 预算 caveat） |
| RL05 | **PARTIAL**（环境/基线/接口/评估完成；RL 未达 Reach 标准） |
| RL06–RL14 | TODO |

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

## RL05 gate 说明（2026-10-16）

- 平台：**真实 Isaac Sim 4.5.0 物理仿真**（RTX 3070 8 GB，`torch.cuda.is_available() == True`）；
  FR3 无手爪模型由双臂码垛工程 URDF 导入为 USD；控制 30 Hz；SB3 2.6.0 经 `--target` 注入，
  未修改 Isaac Sim 安装。
- 门禁全部通过：
  - Jacobian 数值核验 cos_sim **0.9997**（预测 vs 实测末端位移）
  - 经典 DLS 控制器诊断 **24/24 成功**，平均末端误差 0.011 m
  - 环境接口测试 **10/10 PASS**
  - 可达目标池 600（训练）+ 100（评估，不重叠），接受率 93.7%
- 独立评估（100 个固定已验证可达目标；30 Hz；`|ee−target| ≤ 0.02 m` 保持 3 控制步）：

  | Policy | Success | Mean final err (m) | Mean return |
  |---|---|---|---|
  | Random | 0.0 % | 0.2152 | −4.97 |
  | **Classical DLS** | **90.0 %** | **0.0136** | **9.00** |
  | Oracle（myopic 上界参考） | 70.0 % | 0.0326 | 6.46 |
  | SAC（v2, 100k 步） | 12.0 % | 0.0863 | 0.03 |
  | PPO（v2, 150k 步） | 12.0 % | 0.0901 | −0.90 |

- **RL 未达到可重复 Reach 成功标准** → RL05 状态 = **PARTIAL**（环境/经典基线/接口/评估/图表全部完成；
  RL 只学到部分到达行为 0 %→12 %，根因判定为 RL 样本效率/信用分配，已完成一次合规修复尝试）
- 产物：`RL05/`（envs / controllers / scripts / tests / configs / assets / results）、
  `RL05_REPORT.md`、`docs/tasks/RL05_FR3_REACH.md`、`RL05/README.md`

Current gate: RL06（Reward Engineering）之前建议先按 `RL05_REPORT.md` §9 解决并行环境与样本效率问题。
