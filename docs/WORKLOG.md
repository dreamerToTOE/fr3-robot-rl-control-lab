# WORKLOG

Append-only learning and engineering log.

---

## 2026-10-16 — RL05 FR3 Cartesian Reach（真实 Isaac Sim 物理）

### 目标
建立第一套**真实 FR3 物理仿真**的 RL 环境：Actor 输出末端笛卡尔位置增量 → DLS Jacobian/IK → 关节指令。

### 平台
- Ubuntu 22.04.5 / Isaac Sim **4.5.0**（`/home/ubuntu2004/isaacsim-4.5.0/python.sh`，Python 3.10.15）
- GPU：RTX 3070 8 GB（驱动 580 / CUDA 13.0）；Isaac 自带 torch 2.5.1+cu118（CUDA 可用）
- RL：Stable-Baselines3 **2.6.0** 经 `pip install --target RL05/.pydeps` 注入（**未修改 Isaac Sim 安装**；
  SB3 2.9 要求 torch ≥ 2.8，与 Isaac 的 2.5.1 冲突，故降级到 2.6.0）
- FR3 模型：复用双臂码垛工程 `isaac/assets/fr3_no_hand.urdf` → 导入为 `RL05/assets/fr3_no_hand.usd`
  （网格只读引用对方 `franka_description`，**未修改对方仓库任何文件**）

### 步骤与结果
1. **USD 构建 + 接口自省**：`build_fr3_usd.py`、`introspect_and_validate.py`、`benchmark_env.py`
   - Articulation 有效，DOF = `fr3_joint1..7`，EE = `fr3_link7`（Jacobian link index 20）
   - `get_jacobians()` 形状 `(1,21,6,7)`；**Jacobian 数值核验 cos_sim = 0.9997**
   - 吞吐：纯物理 15 000 steps/s；**完整控制步 4.5 ms → 223 步/s**
2. **经典 DLS 诊断 → 发现并修复根因**
   - 首次诊断：24 目标 **0/24 成功**，末端误差全程不变（final ≈ min），关节跟踪误差仅 0.001 rad
   - 根因：任务公式中 `qdot_cmd` 是关节**速度**，`Kp=1.0` 时末端速度只有 0.02 m/s → 每控制步 0.0007 m
   - 修复：`Kp = 1/control_dt = 30 (1/s)` → **24/24 成功**，平均末端误差 0.011 m
   - 统一封装 `controllers/dls_ik.default_fr3_controller()`，所有脚本共用（防止各自调参口径漂移）
3. **可达目标池**：`generate_targets.py` → 600 训练 + 100 评估（不重叠），接受率 **93.7%**
4. **环境 10 项接口测试**：`tests/test_env_interface.py` → **10/10 PASS**
   - 期间拦下一个真实 bug：`reset()` 未给 `self._target` 赋值（(7,) vs (3,)），会导致训练直接崩溃
5. **独立评估（100 个固定可达目标，30 Hz，tol 0.02 m 保持 3 步）**

   | Policy | Success | Mean final err (m) | Mean return |
   |---|---|---|---|
   | Random | 0.0% | 0.2152 | −4.97 |
   | Classical DLS | **90.0%** | **0.0136** | 9.00 |

6. **SAC 30k 步试点**：return −5.0 → −0.56（单调改善）、动作平滑性提升 26 倍，但成功率仅 **2%**，
   末端误差 ≈ 初始距离 → 判定 **训练预算不足**（门禁全过、经典基线同环境 90% → 排除环境/接口/评估问题）
7. **一次修复**：SAC 提升至 **150k** 步、PPO **200k** 步（`sac_seed42_r2` / `ppo_seed42_r2`）
   - 实测吞吐：SAC ≈ 66 env steps/s（每步一次梯度更新）

### 过程中修复的工程缺陷
- SB3 2.6.0 的 SAC 无 `get_params()` → 训练完成后写 metadata 崩溃（改为三级回退）
- 缺 `configs/ppo_fr3_reach.yaml` → PPO 训练静默失败（异常在 try 之外，日志过滤看不到）
- `plot.py` 需用带 matplotlib 的解释器（RL04 venv），不能用系统 `python3`
- 本机 `/tmp` 仅 10 MB → 所有 pip 操作前必须 `TMPDIR=/home/ubuntu2004/.workbuddy/tmp`

### 产出
- `RL05/{envs,controllers,scripts,tests,configs,assets,results}`、`RL05/README.md`、`RL05_REPORT.md`
- 结论见 `RL05_REPORT.md` 与 `results/rl05/comparison/comparison.md`

## 2026-10-15 — RL04 Pendulum Continuous-Control Pipeline

### 目标
第一次完整跑通 train → checkpoint → eval → plot，PPO / SAC 各完成一次真实连续控制训练与独立评估。

### 执行环境
- 本地仓库副本（`/home/ubuntu2004/lmy/fr3-robot-rl-control-lab`）**不含 `.git`**，为部分拷贝，
  故从 GitHub 拉取完整参考副本到 `.upstream_ref/`（commit `05202de2b021c0b53b12acaafb8a9155aa4d00ae`）以读取
  `AGENTS.md` / `CODEX_START_HERE.md` / `docs/*`。
- 独立 venv：`/home/ubuntu2004/.workbuddy/binaries/python/envs/rl04`（Python 3.13.12）。
- 依赖：gymnasium 1.4.0、stable-baselines3 2.9.0、torch 2.14.1+cpu、numpy 2.5.4、matplotlib 3.11.2、pandas 3.0.6。
- 注意：本机 `/tmp` 仅 10MB tmpfs，pip 需 `TMPDIR=/home/ubuntu2004/.workbuddy/tmp`，否则 196MB torch wheel 下载失败。
- 未触碰 ROS2 / Isaac Sim / 系统 Python；`results/RL02_actor_critic_seed42.csv` 等既有数据保持原样。

### 步骤与结果
1. **Smoke test**（`scripts/smoke_test.py`）：obs Box(3,)、action Box(1,)∈[-2,2]、200 步 TimeLimit truncation、
   reward/obs 全 finite → PASS（`results/rl04/smoke_test.json`）。
2. **训练**（`scripts/train.py`，seed=42，50k steps，MlpPolicy 64×64，lr 3e-4，gamma 0.99）：
   - PPO：实际 51200 步，wall 29.5s。
   - SAC：实际 50000 步，wall 210.8s。
3. **独立评估**（`scripts/eval.py`，20 episodes，固定种子 1000..1019，deterministic）：

   | Policy | Mean Return | Std | Mean \|theta\| | Mean \|omega\| | Mean \|u\| | Max \|u\| | Saturation | Upright |
   |---|---|---|---|---|---|---|---|---|
   | Random | -1258.78 | 269.56 | 2.108 | 2.915 | 0.997 | 2.000 | 1.23% | 0.05% |
   | PPO@50k | -1101.13 | 199.94 | 1.769 | 3.454 | 0.275 | 0.781 | 0.00% | 0.75% |
   | SAC@50k | -151.20 | 81.53 | 0.338 | 0.497 | 0.671 | 1.999 | 3.50% | 84.00% |

4. **PPO 失败诊断 → 修复**：
   - 证据：PPO 训练曲线 5 分箱均值 -1177/-1168/-1280/-1160/-1136，无学习趋势；各 checkpoint 评估恒在 -1100~-1235。
   - 排除：环境 bug（smoke PASS）、评估 bug（SAC 同协议成功）、reward/episode 处理 bug（SAC 收敛）。
   - 根因：50k 步下 PPO 仅 25 次 policy update，预算不足。
   - 修复：同超参数扩展至 300k 步 → 评估 mean return **-227.3**，upright **75.0%**；训练曲线 -1186→-291 单调改善。
   - 修复有效，故未再执行备选方案（符合 AGENTS.md anti-loop 规则）。
5. **多随机种子**（43/44）：PPO@50k 三种子均失败（-1152.9 ± 57.2，upright 0.62%）；
   SAC@50k 三种子均成功（-150.6 ± 0.8，upright 83.86%）。
6. **可视化**：`results/rl04/comparison/plots/` 下 9 张图（训练/评估曲线、angle/omega/torque vs time、
   指标柱状、饱和分布、PPO 预算诊断、多随机种子）。
7. **演示视频**：SAC@50k 与 PPO@300k 各 1 个 mp4（headless `SDL_VIDEODRIVER=dummy` 渲染，未影响训练）。

### 工程修复记录（过程中的 bug）
- `train.py` metadata 中 `Path.relative_to` 对相对路径抛 ValueError → 增加 `_rel_to_root` 兜底。
- `eval.py` checkpoint 文件名解析 `rsplit("_",1)` 命中 `steps` 后缀 → 改为正则 `_(\d+)_steps$`。

### 产出
- 代码：`RL04/scripts/{rl04_common,smoke_test,train,eval,plot,record_video,multiseed_summary}.py`、`RL04/run_all.sh`
- 配置：`RL04/configs/{ppo_seed42,sac_seed42,ppo_seed42_extended}.yaml`
- 结果：`RL04/results/rl04/`（每 run 含 config/metadata/final_model/checkpoints/train_metrics/eval_metrics/eval_trace/plots/logs）
- 文档：`RL04_REPORT.md`、本 WORKLOG、`docs/STATUS.md`、`docs/tasks/RL04_PENDULUM_PIPELINE.md`
