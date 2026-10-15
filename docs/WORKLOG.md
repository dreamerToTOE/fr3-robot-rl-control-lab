# WORKLOG

Append-only learning and engineering log.

---

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
