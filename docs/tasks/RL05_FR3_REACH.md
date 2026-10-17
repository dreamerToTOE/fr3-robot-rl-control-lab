# RL05 — FR3 Cartesian Reach

## Goal
创建第一套 FR3 RL 环境。

## Observation
起步建议：q、qdot、EE pose error、previous action。

## Action
优先 Cartesian Δx/Δpose，再由 IK/Jacobian 转为底层命令。

## Reward
距离误差 + success bonus + action/smoothness penalty。

## PASS
固定 seed 集上达到可重复的 reach success，并保存视频/曲线。

---

# 实验记录（2026-10-16 执行）

## 1. 平台与资产（真实 Isaac Sim 物理）

| 项 | 值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS（kernel 6.8.0-138） |
| Isaac Sim | **4.5.0 独立安装** `/home/ubuntu2004/isaacsim-4.5.0`（`./python.sh`，Python 3.10.15） |
| GPU | NVIDIA RTX 3070 8 GB（驱动 580.178.04 / CUDA 13.0） |
| PyTorch | 2.5.1+cu118，`torch.cuda.is_available() == True` |
| FR3 模型 | `isaac/assets/fr3_no_hand.urdf`（复用双臂码垛工程），导入为 `RL05/assets/fr3_no_hand.usd`（16.9 MB） |
| 网格资源 | 只读引用 `dual-arm-embodied-palletizing/ros_ws/install/franka_description`（未修改对方仓库） |
| RL 库 | Stable-Baselines3 **2.6.0**（`--target` 注入 `RL05/.pydeps`，不修改 Isaac 安装） |
| 控制周期 | 30 Hz（`physics_dt=1/120`，`decimation=4`）；单回合 100 控制步 = 3.33 s |
| 并行环境 | 1（单环境，Isaac 物理步进约 15 000 steps/s；完整控制步 4.5 ms → 223 步/s） |

### FR3 关节限位（URDF 实测，保留真实约束）

```text
fr3_joint1  ±2.9007      fr3_joint5  ±2.8763
fr3_joint2  ±1.8361      fr3_joint6  [0.4398, 4.6216]
fr3_joint3  ±2.9007      fr3_joint7  ±3.0508
fr3_joint4  [-3.0770, -0.1169]
effort: j1–j4 = 87 N·m, j5–j7 = 12 N·m;  velocity: 2.62 / 5.26 rad·s⁻¹
```

## 2. MDP 定义

- **Observation（20 维，float32，原始量纲）**：`[q(7), qdot(7), ee_error(3), prev_action(3)]`
  - `q` 关节位置 rad；`qdot` 关节速度 rad/s；`ee_error = target − ee` 世界系 m；`prev_action` 上一步归一化动作
  - 训练时由 `VecNormalize` 归一化（统计量随模型保存为 `vecnormalize_stats.json`），评估时复用同一统计量
- **Action（3 维，∈[-1,1]）**：末端笛卡尔位置增量方向；实际位移 = `action × 0.01 m`／控制步（不直接输出 7 关节力矩）
- **Reward（分项单独记录）**

  ```text
  reward = progress
         + success_bonus        (达成并保持时 +10，仅触发一次)
         - action_penalty       (0.01 · ||a||²)
         - smoothness_penalty   (0.02 · ||a − a_prev||²)
         - safety_penalty       (关节越限 −1)
  progress = |prev_err| − |cur_err|
  ```

- **Termination**：成功 = `|ee − target| ≤ 0.02 m` 连续保持 3 个控制步；失败 = 关节越限；截断 = 100 控制步

## 3. 关键工程修复（均在真实运行中定位）

| # | 问题 | 证据 | 修复 |
|---|---|---|---|
| 1 | **经典 DLS 增益量纲错误（根因）** | 24 目标 0/24 成功，末端误差全程不变（final≈min）；关节跟踪误差仅 0.001 rad → 排除物理/驱动问题 | 任务公式中 `qdot` 是关节**速度**，Kp=1.0 时末端速度仅 0.02 m/s。改为 `Kp = 1/control_dt = 30 (1/s)` → **24/24 成功**，平均误差 0.011 m。统一封装 `controllers/dls_ik.default_fr3_controller()` |
| 2 | `reset()` 未赋值 `self._target` | 环境接口测试第 1 项即崩溃 `(7,) vs (3,)` | `reset()` 中补 `self._target = target.copy()`（门禁测试拦截，避免训练期崩溃） |
| 3 | SB3 2.6.0 的 SAC 无 `get_params()` | 30k 试点训练完成后写 metadata 时崩溃 | 改为 `get_params`/`get_parameters`/配置回退三级 |
| 4 | 缺 `configs/ppo_fr3_reach.yaml` | PPO 训练静默失败（异常在 try 之外） | 补齐配置 |

## 4. 门禁结果

- **Jacobian 数值核验**：`art.get_jacobians()` 形状 `(1,21,6,7)`，EE = `fr3_link7`（index 20）；
  施加 0.05 rad 关节增量后实测末端位移与 `J·dq` 的余弦相似度 **0.9997**
- **经典控制诊断**：24 个目标（8 方向 × 10/20/30 cm）**24/24 成功**，平均末端误差 **0.011 m**，
  耗时随距离线性增长（10 cm ≈ 20 步，30 cm ≈ 80 步），无关节越限
- **可达目标池**：`assets/targets_train.json`（600）+ `assets/targets_eval.json`（100，不重叠），
  接受率 **93.7%**；工作空间 x[0.32, 0.68] y[−0.36, 0.37] z[0.33, 0.95]
- **环境接口测试 `tests/test_env_interface.py`：10/10 PASS**（reset 合法性、观测维度/有限性、
  动作缩放与方向、Jacobian 一致性、reward 分项、成功触发、截断、连续 reset 无漂移、随机策略完整回合）

## 5. 训练预算决策

- 吞吐实测：SAC ≈ 67 env steps/s（含每步梯度更新），PPO ≈ 待测；Isaac 完整控制步 4.5 ms
- 30k 步试点：return 由 −5.0 改善至 −0.56、动作平滑性提升 26 倍，但成功率仅 2%，
  末端误差 ≈ 初始距离 → 初判为训练预算不足（30k 试点保留为诊断证据，`sac_seed42`）
- 进一步诊断发现**奖励尺度失衡**：v1 下 `action_penalty`（0.01·‖a‖²×100 步 ≈ 0.5）大于整个回合
  可用的 progress 信号（≈0.3），"朝目标移动"在即时回报上是负收益 → 存在"少动"局部最优
- 一次修复（v2）：`progress_coef` 1→10、`action_penalty_coef` 0.01→0.001、
  `smoothness_coef` 0.02→0.002（**评估口径完全未改动**）
- 正式训练：**SAC 100k 步（`sac_seed42_v2`，1527 s，65.3 步/s）、
  PPO 150k 步（`ppo_seed42_v2`，842 s，178.8 步/s）**

## 6. 结果

> 见 `RL05_REPORT.md` 与 `results/rl05/comparison/`。评估协议：同一组 100 个已验证可达目标、
> 关节精确复位到 READY、30 Hz、成功判据 `|ee−target| ≤ 0.02 m` 保持 3 控制步、上限 100 控制步。

| Policy | Success Rate | Mean final err (m) | Mean return | 备注 |
|---|---|---|---|---|
| Random | 0.0% | 0.2152 | −4.97 | 参考下界 |
| **Classical DLS** | **90.0%** | **0.0136** | **9.00** | 经典 Baseline，最优 |
| Oracle（myopic） | 70.0% | 0.0326 | 6.46 | 上界参考（证明任务可解） |
| SAC（v2, 100k 步） | 12.0% | 0.0863 | 0.03 | 1500 步/s→65 步/s，平台化 |
| PPO（v2, 150k 步） | 12.0% | 0.0901 | −0.90 | 饱和率 97.3% |

**结论：RL05 = PARTIAL。** 环境 / 经典基线 / 接口门禁 / 评估协议 / 图表全部完成且可复现；
两种 RL 算法都被真实训练并独立评估，但均未达到可重复的 Reach 成功标准（12.0% vs 经典 90.0%）。
失败根因判定为 RL 样本效率/信用分配（单环境、交互量不足），已执行一次合规修复（奖励重平衡，
成功率 0%→12%）后按 anti-loop 规则停止。详见 `RL05_REPORT.md`。
