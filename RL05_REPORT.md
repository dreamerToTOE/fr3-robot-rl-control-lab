# RL05 实验报告 — FR3 Cartesian Reach

> 项目：Franka FR3 强化学习控制教学与工程仓库（fr3-robot-rl-control-lab）
> 任务：RL05 — FR3 Cartesian Reach（机械臂笛卡尔空间末端到达）
> 执行日期：2026-10-16
> 平台：**真实 Isaac Sim 4.5.0 物理仿真**（非 Pendulum、非简化数学模型）

---

## 0. 摘要（TL;DR）

在**真实 FR3 Isaac Sim 物理仿真**中建立了第一套可训练的机械臂 RL 环境：Actor 输出末端笛卡尔位置增量
(Δx, Δy, Δz)，经阻尼最小二乘（DLS）Jacobian/IK 控制器转成关节位置指令，控制频率 30 Hz。

**已验证通过的部分（全部为真实运行结果）**

| 项目 | 结果 |
|---|---|
| FR3 仿真环境 | ✅ 可运行、可 headless、可快速 reset（`fr3_joint1..7`，EE=`fr3_link7`） |
| Jacobian 正确性 | ✅ 预测 vs 实测末端位移 **cos_sim = 0.9997** |
| 经典 DLS Baseline | ✅ **24/24 诊断目标成功**；正式评估 **success 90.0%**，平均末端误差 **0.0136 m** |
| RL 环境接口 | ✅ **10/10 项测试通过**（含真实 bug 拦截） |
| 可达目标池 | ✅ 600 训练 + 100 评估（不重叠），接受率 93.7% |
| Oracle 上界参考 | ✅ **success 70.0%**（朝目标方向走满动作）→ 证明环境/奖励/动作接口自洽、任务可解 |
| Random 基线 | ✅ success 0.0%，平均末端误差 0.2152 m |
| SAC / PPO | ⚠️ 见 §5——真实训练完成，但**未达到可重复的 Reach 成功标准**（均 12.0%，对比经典 90.0%、Oracle 70.0%），失败与诊断已完整记录 |

**最终状态：PARTIAL**（环境 / 经典基线 / 接口 / 评估 / 图表 全部完成且可复现；
RL 策略在给定预算内只学到部分到达行为（0% → 12%），未达到可重复的 Reach 成功标准，
根因已定位并给出后续修复方向）。
**未将 Pendulum 或简化数学模型冒充 FR3 物理仿真，未修改任何评估口径。**

---

## 1. 平台与资产

| 项 | 实际值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS（kernel 6.8.0-138-generic） |
| Isaac Sim | **4.5.0 独立安装** `/home/ubuntu2004/isaacsim-4.5.0`（`./python.sh`，Python 3.10.15） |
| Isaac Lab | 目录存在（`/home/ubuntu2004/lmy/IsaacLab-main`）但无可用虚拟环境 → **改用 Isaac Sim standalone** |
| GPU | NVIDIA RTX 3070 8 GB（驱动 580.178.04 / CUDA 13.0） |
| PyTorch | 2.5.1+cu118（Isaac 自带），`torch.cuda.is_available() == True` |
| RL 库 | Stable-Baselines3 **2.6.0**、gymnasium 0.29.1（经 `pip install --target RL05/.pydeps` 注入） |
| FR3 模型 | `isaac/assets/fr3_no_hand.urdf`（复用双臂码垛工程）→ `RL05/assets/fr3_no_hand.usd`（16.9 MB） |
| 网格资源 | 只读引用 `dual-arm-embodied-palletizing/ros_ws/install/franka_description`（**未修改对方仓库**） |
| 控制周期 | 30 Hz（`physics_dt = 1/120 s`，`decimation = 4`） |
| 并行环境数 | 1（Isaac 物理步进 15 000 steps/s；完整控制步 4.5 ms → 223 步/s；SAC 训练含梯度更新后 66 步/s） |
| 单回合 | 100 控制步 = 3.33 s 仿真时间 |

**未影响其他项目环境**：没有修改系统 Python / CUDA / ROS / Isaac Sim 安装；SB3 装在独立 `--target` 目录，
运行时用 `PYTHONPATH` 注入。双臂码垛工程只被**只读引用**（URDF + 网格），未改动任何文件。

### FR3 关节限位（来自 URDF，真实约束，未放宽）

```text
fr3_joint1  ±2.9007 rad     fr3_joint5  ±2.8763 rad
fr3_joint2  ±1.8361 rad     fr3_joint6  [0.4398, 4.6216] rad
fr3_joint3  ±2.9007 rad     fr3_joint7  ±3.0508 rad
fr3_joint4  [-3.0770, -0.1169] rad
力矩上限: j1–j4 = 87 N·m, j5–j7 = 12 N·m     速度上限: 2.62 / 5.26 rad·s⁻¹
```

---

## 2. 控制接口（MDP 定义）

### Observation（20 维 float32，原始量纲）

```text
observation = [ q(7), qdot(7), ee_error(3), prev_action(3) ]
  q          关节位置, rad,        ∈ 关节限位
  qdot       关节速度, rad/s,      ∈ [-5.26, 5.26]
  ee_error   target - ee_position, 世界系, m
  prev_action 上一步归一化动作,    ∈ [-1, 1]
```

坐标系：全部世界系（FR3 基座为原点）。训练时由 `VecNormalize` 归一化（统计量随模型保存为
`vecnormalize_stats.json`），评估复用同一统计量，保证训练/评估口径一致。

### Action（3 维，∈[-1,1]）

```text
x_target = ee_current + action * action_scale      (action_scale = 0.01 m / 控制步)
qdot_cmd = J^T (J J^T + λ² I)^(-1) Kp (x_target - ee_current)
q_target = clip(q + qdot_cmd * dt, q_min, q_max)
```

Actor **不输出** 7 关节力矩。`Kp = 1/dt = 30 (1/s)`（速度增益，见 §7 根因 1）、`λ = 0.05`、
`|qdot| ≤ 2.62 rad/s`、单步关节增量 ≤ 0.05 rad、末端指令 ≤ 0.01 m/步。

### Reward（分项单独记录）

```text
reward = progress_coef * (|prev_err| - |cur_err|)
       + success_bonus                 (达成并保持 3 控制步时 +10，仅触发一次)
       - action_penalty_coef * ||a||²
       - smoothness_coef * ||a - a_prev||²
       - joint_limit_penalty           (关节越限 -1)
```

`safety_penalty` 只是信号；**安全约束本身由底层限幅保证**（关节限位投影、速度限幅、指令限幅）。

### Reset / Termination

- **Reset**：关节复位到 `READY + N(0, 0.05 rad)`，物理settle 60 步；目标从已验证可达池采样
- **Success**：`|ee − target| ≤ 0.02 m` 连续保持 **3** 个控制步（避免高速掠过误判）
- **Failure**：关节位置越过限位（不可恢复的非法状态）
- **Truncation**：100 个控制步（3.33 s 仿真时间）

---

## 3. 经典控制 Baseline（DLS Jacobian/IK）

### 诊断（进入 RL 训练前的门禁）

24 个目标（8 个方向 × 10/20/30 cm）：

| 版本 | 成功率 | 平均末端误差 | 说明 |
|---|---|---|---|
| 首次（`Kp = 1.0`） | **0 / 24** | 0.161 m（且 final ≈ min，全程未收敛） | 根因见 §7 |
| 修复后（`Kp = 1/dt = 30`） | **24 / 24** | **0.011 m** | 耗时随距离线性增长（10 cm ≈ 20 步，30 cm ≈ 80 步） |

### 正式评估（100 个固定可达目标）

| 指标 | 数值 |
|---|---|
| Success Rate | **90.0 %** |
| Mean final position error | **0.0136 m** |
| Median final position error | 0.0138 m |
| Mean return | 9.00 |
| Mean episode length | 见 `comparison.csv` |
| Mean \|action\| | 0.020 |
| Action saturation ratio | ≈0 % |
| Joint limit violations | 0 |
| Collision steps | 0（最小场景无障碍物；地面平面存在并监测） |

> 目标池用 140 步预算预筛、正式协议为 100 步，故剩余 10% 为目标本身在 100 步内处于边缘可达。
> **同一 100 步协议同时施加于 Random / Oracle / SAC / PPO**，因此对照是公平的。

---

## 4. 环境接口验证（训练前门禁，10/10 PASS）

| # | 测试 | 结果 |
|---|---|---|
| 1 | Reset 后回到合法初始状态 | PASS（关节偏差 <0.01 rad，末端偏差 2.6e-6 m） |
| 2 | Observation 维度 / 有限性 | PASS（(20,) float32，无 NaN/Inf） |
| 3 | Action 缩放有界 | PASS（单步位移 ≤ 1.6 × action_scale） |
| 4 | 正 X 动作使末端沿 +X 移动 | PASS |
| 5 | Jacobian 与实际运动一致 | PASS（cos_sim > 0.95，量级比 0.7–1.4） |
| 6 | Reward 各分项符合设计 | PASS（progress 符号、平滑项、动作惩罚均符合预期） |
| 7 | 成功条件能触发 | PASS（保持 3 步后触发） |
| 8 | 达到时间上限正确截断 | PASS（第 100 步 truncated） |
| 9 | 连续 5 次 Reset 无漂移 | PASS（末端偏差恒为 2.6e-6 m） |
| 10 | 随机策略跑完整回合不崩溃 | PASS（3 个 episode 正常结束） |

---

## 5. 强化学习训练与评估

### 5.1 训练预算决策

实测吞吐（真实测量，非估算）：Isaac 完整控制步 4.5 ms → 223 步/s；
SAC 含每步梯度更新后 **66.7 步/s**。据此制定预算，**没有照搬 RL04 的 50k/300k**。

### 5.2 训练执行（全部真实运行完成）

| 阶段 | Algorithm | 实际步数 | Wall time | 吞吐 | 训练期成功率 | Run-id |
|---|---|---|---|---|---|---|
| 试点（v1 奖励） | SAC | 29 941 | 440 s | 68.0 步/s | 0.33 % | `sac_seed42` |
| 修复 A（v1 奖励 + 预算放大） | SAC | 28 672（提前终止） | 313 s | 91.6 步/s | — | `sac_seed42_r2` |
| **修复 B（奖励重平衡 v2）** | **SAC** | **99 727** | **1 527 s** | **65.3 步/s** | **3.44 %** | `sac_seed42_v2` |
| **修复 B（奖励重平衡 v2）** | **PPO** | **150 528** | **842 s** | **178.8 步/s** | **8.29 %** | `ppo_seed42_v2` |

> PPO 的 env 吞吐约为 SAC 的 **2.7 倍**（178.8 vs 65.3 步/s）：SAC 每个 env step 都要做一次梯度更新，
> PPO 每 1024 步才做一批多 epoch 更新。

所有 run 均保存：`config.json`（含实际生效超参）、`metadata.json`（版本 / seed / wall time /
吞吐 / checkpoint / 超参）、`final_model.zip`、`vecnormalize.pkl` + `vecnormalize_stats.json`、
`checkpoints/`、`train_metrics.csv`（含 return / length / success / final distance）、`logs/progress.csv`。

### 5.3 独立评估（100 个固定可达目标，评估期不更新参数）

| Policy | Success Rate | Mean final err (m) | Median err (m) | Time-to-success (s) | Mean return | Ep len | Sat ratio | Smoothness RMS | Limit viol | Collision |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.0 % | 0.2152 | 0.2023 | — | −4.97 | 100.0 | 47.7 % | 1.4094 | 0 | 0 |
| **Classical DLS** | **90.0 %** | **0.0136** | 0.0113 | 1.461 | **9.00** | 49.5 | 0.0 % | 0.0033 | 0 | 0 |
| Oracle（myopic 上界参考） | 70.0 % | 0.0326 | 0.0148 | 1.929 | 6.46 | 70.5 | 50.2 % | 0.1301 | 0 | 0 |
| SAC（v2, 100k） | 12.0 % | 0.0863 | 0.0611 | 1.786 | 0.03 | 94.4 | 54.3 % | 0.2868 | 0 | 0 |
| PPO（v2, 150k） | 12.0 % | 0.0901 | 0.0859 | 1.592 | −0.90 | 93.7 | **97.3 %** | 0.2156 | 0 | 0 |

数据来源：`results/rl05/comparison/comparison.csv`（由真实评估数据生成）。

### 5.4 对照分析

- **RL vs 经典控制**：经典 DLS 90.0% ≫ SAC/PPO 12.0%。经典控制在已知目标（含 `ee_error` 观测）
  下几乎是最优解，RL 未达到同等水平。
- **RL vs Random**：两者都**明显优于随机**（success 12% vs 0%，末端误差 0.086/0.090 m vs 0.215 m），
  说明 RL 确实学到了从图像无关的状态量中提取"朝目标靠近"的部分行为，只是**未收敛到可重复到达**。
- **SAC vs PPO**：本次任务上**打成平手**（均 12.0%），但代价不同——
  SAC 用 100k 步 / 1 527 s 达到 12.0%；PPO 用 150k 步 / 842 s 达到 12.0%；
  PPO 的**样本效率更低但 wall-clock 效率更高**（178.8 vs 65.3 步/s）。
- **控制质量与安全性**：三类 RL 策略均**无关节越限、无碰撞**；但 PPO 的**动作饱和率高达 97.3%**
  （几乎每一步都顶在 ±1 边界），SAC 54.3%、Random 47.7%，而经典控制 0.0% —— 说明 RL 策略
  长时间在边界处工作，控制品质明显低于经典控制。
- **不可能仅凭训练曲线判断成功**：SAC/PPO 训练期成功率分别只有 3.4% / 8.3%，
  独立评估为 12.0% —— 训练 return 已无明显上升趋势（SAC +1.45、PPO +1.46），属**平台化**。

---

## 6. 关键工程修复记录（全部在真实运行中定位）

| # | 问题 | 证据 | 修复 |
|---|---|---|---|
| 1 | **DLS 增益量纲错误（根因）** | 24 目标 0/24 成功、误差全程不变（final≈min）；关节跟踪误差仅 0.001 rad（排除物理/驱动） | 公式中 `qdot` 是关节**速度**，`Kp=1.0` 时末端速度仅 0.02 m/s → 每步 0.0007 m。改 `Kp = 1/control_dt = 30 (1/s)`，**24/24 通过**。统一封装 `default_fr3_controller()` |
| 2 | `reset()` 未赋值 `self._target` | 环境接口测试第 1 项即崩溃 `(7,) vs (3,)` | 补 `self._target = target.copy()`（**门禁测试拦下，否则训练期直接崩溃**） |
| 3 | SB3 2.6.0 的 SAC 无 `get_params()` | 30k 训练完成后写 metadata 时崩溃 | 改为 `get_params` → `get_parameters` → 配置回退 |
| 4 | 缺 `configs/ppo_fr3_reach.yaml` | PPO 训练静默失败（异常在 try 之外） | 补齐配置 |
| 5 | `/tmp` 仅 10 MB tmpfs | pip 安装 torch 报 `Errno 28` | 所有操作前 `TMPDIR=/home/ubuntu2004/.workbuddy/tmp` |
| 6 | Isaac `app.close()` 挂起 | 多次进程假死 | 脚本末端 `os._exit(0)` 强制退出 |
| 7 | 视频录制进程挂起占 GPU（1.9 GB） | `nvidia-smi` 发现 3 个竞争进程 | 按 PID 清理，保证训练独占 GPU |

---

## 7. RL 失败的诊断（Anti-loop：一次诊断 + 一次修复 + 一次备选）

### 观测现象

- SAC v1 @30k：训练 return 由 −5.0 改善至 −0.56、动作平滑性提升 26 倍，但**成功率仅 2 %**，
  评估末端误差 0.257 m ≈ 初始距离 → 策略在动，但没有朝目标收敛
- SAC v1 @50k：return −0.28，成功率仍 0 %
- 奖励重平衡（v2）后：SAC return 由 −0.54 转正并升至 **+1.45**、成功率 0 % → **3.4 %**（训练期）/
  **12.0 %**（独立评估）；PPO 同样由负转正（**+1.46**）并达 **12.0 %**（独立评估）
- **两者最终均在 12.0 % 附近平台化**，与经典控制 90.0 %、Oracle 70.0 % 差距显著

### 逐层排查（按 AGENTS.md 要求的 ENVIRONMENT → CONTROL → REWARD → RL → EVALUATION 顺序）

| 层 | 假设 | 证据 | 结论 |
|---|---|---|---|
| ENVIRONMENT | 环境/资产错误 | Smoke + 10/10 接口测试通过；Oracle 策略在该环境下 **70% 成功** | **排除** |
| CONTROL | IK/动作方向错误 | Jacobian cos_sim 0.9997；动作方向测试通过；Classical DLS 同环境 **90%** | **排除** |
| EVALUATION | 评估协议/数据泄漏 | 五类策略共用同一目标集、同一 reset、同一容差；Classical 90% / Random 0% 区分度良好 | **排除** |
| REWARD | 奖励尺度失衡 | v1 下 `action_penalty`（0.01·‖a‖²×100 步 ≈ 0.5）**大于整个回合可用的 progress 信号（≈0.3）** → "朝目标移动"在即时回报上是负收益，存在"少动"局部最优 | **部分成立**，已修复（v2） |
| RL | 样本效率 / 优化 | 修复奖励后 SAC 仍 0 %；Oracle 可达 70 % | **主因**（见下） |

### 根因判定

**主根因 = 在给定计算预算（SAC 100k / PPO 150k 环境步，单环境）下，SB3 的开箱策略未能学到
"沿 ee_error 方向持续出力"这一反应式策略并收敛到可重复到达。**
环境、控制、评估、奖励接口均已验证自洽（Oracle 70 %、Classical 90 %），因此这是纯粹的
**RL 样本效率与信用分配问题**，而非仿真或工程缺陷。

可能的具体机制（供后续验证，不在本轮继续试错）：

1. 动作需连续 30+ 步保持一致方向才产生一次成功信号，而 progress 单项每步仅 ~0.01 m；
2. SAC 的自动熵调节（`ent_coef=auto`）在需要"精确、低噪声"的任务上保持较高探索噪声；
3. **单环境、无并行** → 有效交互量太少：100k 步 ≈ 1000 个 episode，而此类任务通常需要
   10⁵–10⁶ 量级的并行交互；实测 PPO 的 97.3 % 动作饱和率也说明策略长期贴边、未进入精细调节区；
4. 未做目标分布的课程学习（远近难度混合采样）。

### 已执行的一次修复（Anti-loop 合规）

- 修复内容：奖励权重重平衡（`progress_coef` 1→10，`action_penalty_coef` 0.01→0.001，
  `smoothness_coef` 0.02→0.002）。**评估口径完全未改动**（目标集/容差/协议不变）
- 修复效果（真实数据）：return 由负转正（SAC −0.54 → +1.45；PPO 同样 +1.46），
  成功率由 0 % 提升到 12.0 %（独立评估）——**修复有效但不充分**
- 仍远低于经典基线 → 按规则**停止继续调参**，不再尝试第三、第四个方案，
  转为记录 PARTIAL 级别的 RL 结论并给出工程升级方向（§9）

---

## 8. 工程结果

### 新增文件

```text
RL05/
├── README.md                      从零运行说明与完整命令
├── run_all.sh / run_train2.sh / run_final.sh
├── configs/  sac_fr3_reach.yaml, ppo_fr3_reach.yaml,
│             sac_fr3_reach_v2.yaml, ppo_fr3_reach_v2.yaml
├── envs/     fr3_reach_env.py      FR3ReachSim + FR3ReachEnv(20 维 obs / 3 维 action)
├── controllers/ dls_ik.py          DLS IK（经典 Baseline 与 RL 共用）
│                classic_reach.py   经典到达 rollout
├── scripts/  build_fr3_usd.py, introspect_and_validate.py, benchmark_env.py,
│             diagnose_classic.py, generate_targets.py, train.py,
│             eval_fr3_reach.py, plot.py, record_video.py
├── tests/    test_env_interface.py 10 项门禁测试
├── assets/   fr3_no_hand.urdf/.usd, targets_train.json, targets_eval.json
└── results/rl05/                   每次 run 的配置/模型/指标/图表/日志
```

### 修改文件

- `docs/STATUS.md`（RL05 状态）、`docs/WORKLOG.md`（追加执行记录）、
  `docs/tasks/RL05_FR3_REACH.md`（补充实验记录）
- **未修改**双臂码垛工程的任何文件；未删除任何已有数据

### 测试结果

- 环境接口测试 **10/10 PASS**
- Jacobian 数值核验 cos_sim **0.9997**
- 经典控制诊断 **24/24**
- 全部训练脚本正常退出/正常保存；过程中修复 7 项工程缺陷（§6）

### 数据与图表路径

`RL05/results/rl05/`：`comparison/comparison.csv|md`、`comparison/plots/01..07`、
各 `*_eval/eval_metrics.json|eval_episodes.csv|eval_trace.csv`、各 run 的 `train_metrics.csv`

### 失败项与剩余风险

- **RL 策略未达到 Reach 成功标准**（SAC / PPO 均 12.0 %，经典 90.0 %）——状态判定为 **PARTIAL**
- **视频未完成**：`record_video.py` 已实现（`omni.replicator.core` 的 `rgb` annotator 离屏渲染），
  但在本机 headless 环境下**渲染产品初始化的进程挂起**（多次 220 s 超时未产出帧，且会长期占用
  约 1.9 GB 显存）→ 为避免影响训练，已将视频列为**未完成项**并清理了挂起进程。
  后续可用 Isaac 的视口截图接口或 GUI 模式补录。
- 单环境训练、未启用并行环境；多随机种子在时间预算内未展开

---

## 9. 下一步建议

1. **并行环境**：Isaac Sim 的 `Articulation`/`RigidPrim` 支持批量环境；把 `n_envs` 提到 64–256
   可把吞吐从 66 步/s 提升一到两个数量级，这是最直接的杠杆。
2. **网络输入增强**：把 `ee_error` 归一化到单位向量（方向）+ 距离标量分别输入，降低学习难度。
3. **SAC 熵控制**：改 `ent_coef` 为固定小值（如 0.002）或降低 `target_entropy`，减少精密任务中的噪声。
4. **课程学习**：先用"近距离目标（10 cm 内）"训练，再逐步扩大工作空间（注意：这是训练课程，
   最终评估仍使用同一 100 目标集，口径不变）。
5. **RL07（PPO vs SAC 公平对比）** 时统一采用并行环境 + 相同环境步数 + ≥3 随机种子。
