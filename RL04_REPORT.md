# RL04 实验报告 — Pendulum 连续控制流水线

> 项目：Franka FR3 强化学习控制教学与工程仓库（fr3-robot-rl-control-lab）
> 任务：RL04 — Pendulum Continuous-Control Pipeline
> 执行日期：2026-10-15
> 执行者：WorkBuddy（Agent 模式）

---

## 0. 摘要（TL;DR）

在 `Pendulum-v1` 上使用 Gymnasium 1.4.0 + Stable-Baselines3 2.9.0 + PyTorch 2.14.1(CPU) 完成了 PPO 与 SAC 的真实训练、独立评估与可视化，并给出了可复现的完整流水线。

- **SAC 在 50k 步内即稳定学会摆杆起摆控制**：独立评估 mean return **-151.2**（random 基线 -1258.8），upright fraction **84.0%**。
- **PPO 在 prescribed 50k 步内未学会**（mean return -1101.1，upright 0.75%，仅略优于 random）。
- 经诊断确认根因是**训练预算/更新次数不足**（而非环境、评估或代码缺陷），**扩展至 300k 步后 PPO 学会**（mean return -227.3，upright 75.0%）。
- 3 个随机种子（42/43/44）复核：SAC@50k 稳定成功（-150.6 ± 0.8），PPO@50k 稳定失败（-1152.9 ± 57.2）。

**最终状态：PASS**（流水线、评估、可视化、代码、可复现性全部达标；附"PPO 在 50k 内失败"的明确 caveat）。

---

## 1. 执行前检查

### 1.1 仓库定位
- 工作目录：`/home/ubuntu2004/lmy/fr3-robot-rl-control-lab/RL04`
- **本地副本不是 git 仓库**（无 `.git`），且为部分拷贝：仅含 `results/RL02_actor_critic_seed42.csv`、`RL02_actor_critic_lab/`、`scripts/rl01_five_state_mdp.py`，`RL04/` 为空目录。
- 因此从 GitHub 拉取完整参考副本到仓库根 `.upstream_ref/`（`--depth 1`），commit `05202de2b021c0b53b12acaafb8a9155aa4d00ae`，用于阅读规范文档。

### 1.2 已阅读文档
`README.md`、`AGENTS.md`、`CODEX_START_HERE.md`、`docs/STATUS.md`、`docs/tasks/RL04_PENDULUM_PIPELINE.md`、`docs/EXPERIMENT_PROTOCOL.md`、`docs/ENGINEERING_RULES.md`。

上游 `docs/STATUS.md` 记录 RL00=READY、RL01–RL14=TODO；本地已有 RL01/RL02 产物。

### 1.3 已有 RL04 成果检查
本地 `RL04/` 目录为空，**不存在** `rl04_pendulum_lab.zip`，也无既有 RL04 代码/checkpoint/结果 → 按指令从零自行实现，未覆盖任何已有成果。

### 1.4 Git 与未提交修改
本地仓库无 `.git`，无法执行 `git status`；未做任何推送或远程分支修改。

---

## 2. 环境搭建

为避免影响 Isaac Sim / ROS2 Humble 等项目，使用**独立 venv**，未改动系统 Python / CUDA / ROS。

```bash
/home/ubuntu2004/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
    -m venv /home/ubuntu2004/.workbuddy/binaries/python/envs/rl04
export TMPDIR=/home/ubuntu2004/.workbuddy/tmp      # 关键：本机 /tmp 仅 10MB tmpfs
```

**实际安装版本（最终生效）**

| 组件 | 版本 |
|---|---|
| Python | 3.13.12 |
| gymnasium | 1.4.0 |
| stable-baselines3 | 2.9.0 |
| torch | 2.14.1+cpu |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| pandas | 3.0.6 |
| pyyaml / imageio / imageio-ffmpeg / pygame-ce | 6.0.3 / 2.38.1 / 0.6.0 / 2.5.8 |
| CPU / GPU | 16 核 CPU / 无 CUDA（`cuda_available=false`） |

> 遇到的环境问题：`/tmp` 为 10MB tmpfs，首次 pip 安装 196MB 的 torch wheel 报 `Errno 28 设备上没有空间`。
> 通过把 `TMPDIR` 指向大磁盘目录解决（记录于 `run_all.sh` 头部与 WORKLOG）。

### 2.1 Smoke Test（必须通过，否则禁止训练）

`scripts/smoke_test.py` → `results/rl04/smoke_test.json`：

| 检查项 | 结果 |
|---|---|
| 创建 / reset | ✅ |
| observation 空间 | `Box([-1,-1,-8], [1,1,8], (3,), float32)` ✅ |
| action 空间 | `Box(-2.0, 2.0, (1,), float32)` ✅ |
| `step()` 返回 5 元组 | ✅ |
| reward / obs NaN·Inf | 全 finite ✅ |
| 完整 episode | 200 步，TimeLimit truncation=True ✅ |
| 随机策略 episode | return -883.2，全 finite ✅ |

**结论：Smoke Test PASS。**

---

## 3. 训练配置与执行

两种算法**完全独立**训练，环境统一 `Pendulum-v1`，`MlpPolicy`，网络 `64×64`，`gamma=0.99`，`lr=3e-4`，`seed=42`。

| 参数 | PPO | SAC |
|---|---|---|
| Total timesteps | 50 000 | 50 000 |
| n_steps / buffer | n_steps=2048, batch=64, n_epochs=10 | buffer=100000, batch=256 |
| 其他 | clip_range=0.2, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5 | tau=0.005, train_freq=1, gradient_steps=1, learning_starts=100, ent_coef=auto |
| Checkpoint | 每 10 000 步 | 每 10 000 步 |
| 实际训练步数 | **51 200**（SB3 按 n_steps 取整） | **50 000** |
| Wall time | 29.5 s | 210.8 s |

> 额外实验（诊断用）：PPO 超参数不变、预算扩展至 **300k** 步（实际 301 056 步，174.5 s）。
> 多随机种子：PPO/SAC 各补 seed 43、44（各 50k 步）。

每个 run 独立目录，含 `config.json`、`metadata.json`（含 git commit / 版本 / seed / wall time / checkpoint 列表）、`final_model.zip`、`checkpoints/`、`train_metrics.csv`、`logs/`。

---

## 4. 独立评估（训练与评估严格分离）

**评估协议**（Random / PPO / SAC 完全一致）

- 20 个独立 episode，固定种子 `1000..1019`（`env.reset(seed=...)` 决定初始状态分布）
- deterministic 动作（Random 用 `env.action_space.sample()`）
- 每 episode 固定 200 步 TimeLimit；**truncation 不视为失败**
- 评估过程中**不更新任何网络参数**

**容差（显式记录）**：upright = `|theta| ≤ 0.2 rad` 且 `|theta_dot| ≤ 1.0 rad/s`；saturation = `|u| ≥ 0.99 × 2.0 N·m`。

### 4.1 seed=42 主结果

| Policy | Mean Return | Std | Episode Len | Mean \|theta\| (rad) | Mean \|omega\| (rad/s) | Mean \|u\| (N·m) | Max \|u\| (N·m) | Saturation | Upright |
|---|---|---|---|---|---|---|---|---|---|
| Random | -1258.78 | 269.56 | 200 | 2.108 | 2.915 | 0.997 | 2.000 | 1.23% | 0.05% |
| PPO @50k | -1101.13 | 199.94 | 200 | 1.769 | 3.454 | 0.275 | 0.781 | 0.00% | 0.75% |
| **SAC @50k** | **-151.20** | 81.53 | 200 | **0.338** | 0.497 | 0.671 | 1.999 | 3.50% | **84.00%** |
| PPO @300k | -227.27 | 133.86 | 200 | 0.470 | — | — | — | 7.38% | 75.00% |

### 4.2 多随机种子（50k 步）

| Algo | Seed 42 | Seed 43 | Seed 44 | 均值 ± 标准差 | Upright（均值） |
|---|---|---|---|---|---|
| PPO | -1101.1 | -1143.2 | -1214.3 | **-1152.9 ± 57.2** | **0.62%** |
| SAC | -151.2 | -149.7 | -150.8 | **-150.6 ± 0.8** | **83.86%** |

### 4.3 样本效率（checkpoint 评估）

| 训练步数 | 50k | 100k | 150k | 200k | 250k | 300k |
|---|---|---|---|---|---|---|
| PPO | -1102 | -964 | -923 | -428 | -288 | -227 |
| SAC | -158 (10k) | -150 | -150 | -151 | — | — |

- **SAC 在 10k 步即达 -158**；PPO 到 200k 步才到 -428，样本效率差距约 **一个数量级**。
- PPO 在 10k/20k/30k/40k/50k 各 checkpoint 评估为 -1235/-1198/-1208/-1171/-1102，**基本平坦**。

---

## 5. 可视化产物

`RL04/results/rl04/comparison/plots/`（全部来自真实记录数据，无人工构造）：

| 文件 | 内容 |
|---|---|
| `01_training_return_vs_steps.png` | Training Return vs Environment Steps（PPO/SAC，含滑动平均） |
| `02_eval_return_vs_steps.png` | Evaluation Return vs Steps（每 checkpoint，20-ep 评估） |
| `03_angle_error_vs_time.png` | \|Angle Error\| vs Time（Random/PPO/SAC，均值±标准差带） |
| `04_angular_velocity_vs_time.png` | \|Angular Velocity\| vs Time |
| `05_torque_vs_time.png` | \|Torque\| vs Time |
| `06_comparison_bars.png` | Random vs PPO vs SAC 指标对照柱状图 |
| `07_action_saturation.png` | 动作饱和率 + torque 分布 |
| `08_ppo_budget_diagnosis.png` | PPO 50k vs 300k 预算诊断 |
| `09_multiseed_return.png` | 多随机种子对照 |

另外每个 run 目录下 `plots/{train_return_vs_steps,eval_return_vs_steps}.png`；演示视频 2 个 mp4（SAC@50k、PPO@300k，headless 渲染）。

---

## 6. 结果分析（逐条回答验收 7 问）

**Q1. PPO 和 SAC 是否都完成了训练？**
是。PPO 实际 51 200 步（29.5 s），SAC 实际 50 000 步（210.8 s），各自保存了 final model 与 5 个 checkpoint。额外完成 PPO 300k 与 4 个多随机种子训练。

**Q2. 两种策略是否比 Random Policy 表现更好？**
SAC 显著更好（-151.2 vs -1258.8，提升约 1108）。PPO@50k 仅微弱更好（-1101.1 vs -1258.8，提升 157），upright 仅 0.75% vs 0.05%，**实质上未学会**。PPO@300k 显著更好（-227.3）。

**Q3. 两者累计 Reward 的变化趋势？**
SAC：前 10k 步快速上升（-845 → -140 区间），随后稳定在 -150 平台，3 个种子方差极小（±0.8）。PPO@50k：全程平坦在 -1100~-1280，无上升趋势。PPO@300k：单调上升（-1186 → -1088 → -984 → -840 → -439 → -291）。

**Q4. 两者是否真正学会控制摆杆接近竖直？**
SAC **学会**：upright 84.0%，mean |theta| 0.338 rad（≈19°），从随机初始摆角起摆并稳定。PPO@50k **未学会**：upright 0.75%，mean |theta| 1.769 rad（≈101°），接近随机。PPO@300k **学会**：upright 75.0%，mean |theta| 0.470 rad。

**Q5. 是否出现力矩长期饱和、振荡或无法收敛？**
SAC 无长期饱和（饱和率 3.5%，均值 |u| 0.671 N·m，max 1.999 N·m 偶发触顶）。PPO@50k 动作幅度极小（|u| 0.275 N·m、饱和率 0%），是"未学习导致的乏力"而非饱和。PPO@300k 饱和率 7.4% 略高、评估 std 较大（133.9），说明策略仍带噪声、尚未完全收敛。两者均未出现 `NaN` 或发散。

**Q6. 环境样本效率、训练耗时区别？**
- 样本效率：**SAC ≫ PPO**。SAC 10k 步收敛；PPO 需 ≥200k 步才明显起效。
- 训练耗时：CPU 单环境下 SAC 50k≈211 s，PPO 50k≈30 s，SAC 约为 PPO 的 **7 倍**（SAC 每步做一次梯度更新并频繁重放采样；PPO 每 2048 步才做一次多 epoch 批量更新）。

**Q7. 哪些结论可确认，哪些需要更多种子/预算？**
- **可确认（3 种子）**：SAC@50k 稳定成功且方差极小；PPO@50k 稳定失败；SAC 样本效率显著优于 PPO。
- **需更多种子/预算才能确认**：PPO 成功所需的精确预算阈值（介于 150k–200k 之间）；PPO 的最终性能上界与稳定性（当前 300k 仍在改善、std 较大）；加入 `VecNormalize` / 多并行环境后 PPO 能否在 50k 内成功。

---

## 7. 失败分析（PPO@50k，根因定位）

按 `AGENTS.md` §4「失败必须先区分 ENVIRONMENT / CONTROL / RL / EVALUATION」逐层排除：

| 假设 | 证据 | 结论 |
|---|---|---|
| ENVIRONMENT 接口错误 | Smoke Test 全通过；SAC 使用**同一** `make_env` 成功 | 排除 |
| EVALUATION 协议错误 | Random/PPO/SAC 使用**完全相同**评估协议与种子；SAC 得 84% upright | 排除 |
| REWARD / Episode 处理错误 | reward 由 Gymnasium 原样给出，未做任何包装改动；SAC 收敛到接近最优 return | 排除 |
| 模型/算法参数错误 | 检查更新次数：50k 步 ÷ n_steps=2048 = **仅 25 次 PPO 更新**；ent_coef=0、单环境 | 部分成立（非参数"错"，而是更新太少） |
| **训练预算不足** | 超参数不变扩至 300k → 学会（-227.3，upright 75%）；曲线单调上升 | **根因成立** |

**处置**：按 anti-loop 规则执行 **一次诊断 + 一次修复**。修复（扩展预算）有效，故**未启用备选方案**。全程未修改评估标准，未伪造任何结果。

---

## 8. 工程结果

### 8.1 新增文件

```
fr3-robot-rl-control-lab/
├── .gitignore                                    # 新增（镜像上游规则）
├── .upstream_ref/                                # 新增：GitHub 完整参考副本 @05202de2
├── RL04_REPORT.md                                # 新增：本报告
├── docs/
│   ├── STATUS.md                                 # 新增：RL04 状态
│   ├── WORKLOG.md                                # 新增：执行记录
│   └── tasks/RL04_PENDULUM_PIPELINE.md           # 新增：实验记录
└── RL04/
    ├── run_all.sh
    ├── configs/{ppo_seed42,sac_seed42,ppo_seed42_extended}.yaml
    ├── scripts/{rl04_common,smoke_test,train,eval,plot,record_video,multiseed_summary}.py
    └── results/rl04/...                          # 见 §3/§4
```

### 8.2 修改文件
**无。** 未修改其他 Task 的代码，未删除任何已有数据（`results/RL02_actor_critic_seed42.csv` 等保持原样）。

### 8.3 测试结果
- Smoke Test：**PASS**
- 训练脚本：7 个 run 全部正常退出（PPO×4、SAC×3）
- 评估脚本：8 次独立评估（random/ppo/sac × seed42 + ppo/sac sweep + ppo300k）+ 4 次多种子评估，全部成功
- 过程中修复 2 个自身脚本 bug（`train.py` 相对路径 `relative_to`、`eval.py` checkpoint 名解析），修复后重跑通过

### 8.4 数据与图表路径
`RL04/results/rl04/`（比对表：`comparison/comparison.csv`、`comparison/multiseed_summary.csv`，图表：`comparison/plots/01..09`）

### 8.5 失败项与剩余风险
- 失败项：PPO 在 prescribed 50k 预算内未学会（已诊断+修复，记录在案）。
- 剩余风险：PPO 300k 结果仅单种子（42），其收敛稳定性未做多种子确认；`VecNormalize` 未启用；训练为单环境 `n_envs=1`。
- 环境风险：本机 `/tmp` 仅 10MB 且 clone 到的本地仓库无 `.git`，后续任务需沿用 `TMPDIR` 覆盖策略。

---

## 9. 下一步建议

1. **进入 RL05（FR3 Reach）**：流水线已验证可用（config→train→checkpoint→eval→plot 全自动），可直接迁移到 FR3 环境。
2. 在 RL05 起启用 **observation normalization（`VecNormalize`）** 与 **多并行环境**，这两项通常能让 PPO 的样本效率提升数倍，避免重复 RL04 的"预算不足"陷阱。
3. 建议把 RL04 的 PPO 预算结论（50k 不足 → 200k+ 起效）作为 RL05/RL07 的默认基线设定。
4. 若要严谨对比 PPO vs SAC，应在 RL07 用 **≥3 种子 × 相同环境步数** 并同时报告 wall-clock 与样本效率。

---

## 附录 A：复现命令（最小集）

```bash
export TMPDIR=/home/ubuntu2004/.workbuddy/tmp
cd /home/ubuntu2004/lmy/fr3-robot-rl-control-lab/RL04
PY=/home/ubuntu2004/.workbuddy/binaries/python/envs/rl04/bin/python
$PY scripts/smoke_test.py
$PY scripts/train.py --algo ppo --config configs/ppo_seed42.yaml
$PY scripts/train.py --algo sac --config configs/sac_seed42.yaml
$PY scripts/eval.py --algo random --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo ppo    --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo sac    --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo ppo --mode sweep --episodes 20 --seed-base 1000
$PY scripts/eval.py --algo sac --mode sweep --episodes 20 --seed-base 1000
$PY scripts/plot.py
$PY scripts/multiseed_summary.py
# 或一键：bash run_all.sh
```

## 附录 B：环境版本快照

见 `RL04/results/rl04/ppo_seed42/metadata.json` 的 `versions` 字段；上游参考 commit `05202de2b021c0b53b12acaafb8a9155aa4d00ae`。
