# CODEX_START_HERE.md

## Mission
建立一套面向 FR3 / 双 FR3 的强化学习运控学习与研究栈：先掌握必要 RL 基础，再通过连续控制、接触操作、Residual RL、安全约束和双臂协同逐步进入研究级工程。

## 当前阶段
当前从 **RL00 — MDP & Robot RL Formulation** 开始。

## 核心策略
- 15% 理论，85% 工程。
- 先高层 action，后 torque action。
- 先单臂，后双臂。
- 先 classical baseline，再 RL，再 residual / safe RL。
- 每个 Task 必须可复现、可评估、可解释。

## 默认平台
- 基础连续控制：Gymnasium 或等价轻量环境。
- 机器人主平台：Isaac Lab / Isaac Sim。
- FR3 控制层：优先 Cartesian delta / velocity / controller-parameter action。
- 传统控制复用：IK、Jacobian、Admittance、QP、安全过滤。

## 每次 Agent 执行前
读取 `AGENTS.md`、`docs/STATUS.md`、当前 Task、`docs/EXPERIMENT_PROTOCOL.md`。
