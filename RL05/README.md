# RL05 — FR3 Cartesian Reach

第一套**真实 FR3 物理仿真**的强化学习环境：Actor 输出末端笛卡尔位置增量（Δx, Δy, Δz），
经阻尼最小二乘（Damped Least Squares, DLS）Jacobian/IK 控制器转成关节位置指令，在 Isaac Sim 中执行。

---

## 1. 环境要求（本机已具备）

| 组件 | 本机实际值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS，kernel 6.8.0-138 |
| Isaac Sim | **4.5.0** 独立安装 `/home/ubuntu2004/isaacsim-4.5.0`（`./python.sh`，Python 3.10.15） |
| GPU | NVIDIA RTX 3070 8 GB（驱动 580 / CUDA 13.0） |
| PyTorch | 2.5.1+cu118（Isaac 自带，`torch.cuda.is_available() == True`） |
| 机器人模型 | FR3 无手爪 URDF（15 830 B），由双臂码垛工程 `isaac/assets/fr3_no_hand.urdf` 复用 |
| 网格资源 | 只读引用 `dual-arm-embodied-palletizing/ros_ws/install/franka_description/share`（未修改对方仓库） |

> ⚠️ 本机 `/tmp` 仅 10 MB tmpfs，任何 pip / 大临时文件操作前必须
> `export TMPDIR=/home/ubuntu2004/.workbuddy/tmp`，否则报 `Errno 28 设备上没有空间`。

## 2. 一次性准备（约 1 分钟）

```bash
export TMPDIR=/home/ubuntu2004/.workbuddy/tmp

# ① 把 Stable-Baselines3 注入独立目录（不修改 Isaac Sim 安装）
RL05=/home/ubuntu2004/lmy/fr3-robot-rl-control-lab/RL05
mkdir -p $RL05/.pydeps
/home/ubuntu2004/isaacsim-4.5.0/python.sh -m pip install \
    --target $RL05/.pydeps --no-deps --timeout 90 --retries 6 \
    stable-baselines3==2.6.0 cloudpickle pandas

# ② 由 URDF 构建 USD 资产（约 10 秒）
cd /home/ubuntu2004/isaacsim-4.5.0
./python.sh $RL05/scripts/build_fr3_usd.py
```

> 选 SB3 **2.6.0** 的原因：Isaac Sim 4.5 自带 torch 2.5.1，而 SB3 2.9 要求 torch ≥ 2.8；
> 2.6.0 只要求 ≥ 2.3，且与 gymnasium 0.29.1 兼容。

## 3. 完整流程（从零复现）

```bash
export TMPDIR=/home/ubuntu2004/.workbuddy/tmp
RL05=/home/ubuntu2004/lmy/fr3-robot-rl-control-lab/RL05
cd /home/ubuntu2004/isaacsim-4.5.0
export PYTHONPATH=$RL05/.pydeps

# ---- 门禁：接口与运动学验证 ----
./python.sh $RL05/scripts/benchmark_env.py               # 单步开销 / 吞吐
./python.sh $RL05/scripts/diagnose_classic.py            # 经典控制器诊断（24 目标，要求全通过）
./python.sh $RL05/scripts/generate_targets.py --train 600 --eval 100   # 生成已验证可达目标池
./python.sh $RL05/tests/test_env_interface.py            # 环境 10 项接口测试（必须全过）

# ---- 训练 ----
./python.sh $RL05/scripts/train.py --algo sac --config $RL05/configs/sac_fr3_reach.yaml
./python.sh $RL05/scripts/train.py --algo ppo --config $RL05/configs/ppo_fr3_reach.yaml

# ---- 独立评估（100 个固定可达目标）----
./python.sh $RL05/scripts/eval_fr3_reach.py --policy classical
./python.sh $RL05/scripts/eval_fr3_reach.py --policy random
./python.sh $RL05/scripts/eval_fr3_reach.py --policy sac --run-id sac_seed42
./python.sh $RL05/scripts/eval_fr3_reach.py --policy ppo --run-id ppo_seed42

# ---- 图表与视频 ----
python3 $RL05/scripts/plot.py
./python.sh $RL05/scripts/record_video.py --policy sac --run-id sac_seed42
```

## 4. 目录结构

```text
RL05/
├── configs/     sac_fr3_reach.yaml, ppo_fr3_reach.yaml
├── envs/        fr3_reach_env.py        # Gymnasium 环境 + FR3ReachSim
├── controllers/ dls_ik.py               # 阻尼最小二乘 IK（经典 Baseline 与 RL 共用）
│                classic_reach.py        # 经典到达 rollout
├── scripts/     build_fr3_usd.py        # URDF → USD
│                benchmark_env.py        # 吞吐/单步开销
│                diagnose_classic.py     # 经典控制器诊断
│                generate_targets.py     # 可达目标池生成与验证
│                train.py / eval_fr3_reach.py / plot.py / record_video.py
├── tests/       test_env_interface.py   # 10 项接口测试
├── assets/      fr3_no_hand.urdf/.usd, targets_train.json, targets_eval.json
└── results/rl05/                        # 每次 run 的 config/metadata/模型/指标/图表/视频
```

## 5. 控制接口摘要

- **Observation（20 维）** `[q(7), qdot(7), ee_error(3), prev_action(3)]`，原始量纲，由 `VecNormalize` 归一化
- **Action（3 维）** `∈[-1,1]`，实际末端位移 = `action × 0.01 m`／控制步
- **控制频率** 30 Hz（`physics_dt=1/120`，`decimation=4`），单回合上限 100 控制步（3.33 s）
- **成功判据** `|ee − target| ≤ 0.02 m` 连续保持 3 个控制步
- **Reward** `progress + success_bonus − action_penalty − smoothness_penalty − safety_penalty`（分项单独记录）
