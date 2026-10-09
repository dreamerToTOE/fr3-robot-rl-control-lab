# AGENTS.md — Codex / Agent Mandatory Rules

本仓库是“教学 + 工程 + 可复现实验”项目。目标不是让 Agent 一次性把全部 RL 系统写完，而是让用户能够逐步理解、运行、比较并迭代。

## 1. 每次开始任务前必须读取
1. `CODEX_START_HERE.md`
2. `docs/STATUS.md`
3. 当前 `docs/tasks/RLxx_*.md`
4. `docs/EXPERIMENT_PROTOCOL.md`（涉及训练/评估时）
5. `git status`

先输出：

```text
=== PRE-TASK REPORT ===
Task:
Learning objective:
Engineering objective:
What the user should understand after this task:
Observation:
Action:
Reward:
Termination:
Baseline:
Expected files:
Validation plan:
Known risks:
Need user confirmation: yes/no
```

## 2. 教学优先级
- 先解释“为什么”，再实现“怎么做”。
- 公式必须对应到机器人中的具体变量。
- 英文术语首次出现时必须给中文解释。
- 不要一次引入超过当前任务需要的 RL 理论。
- 用户已有机器人运控基础，可直接使用 Jacobian、IK、阻抗/导纳、QP、操作空间等术语，但遇到 RL 新概念必须解释。

## 3. 工程优先级
- RL04 以后，每个任务必须产生真实可运行代码或可验证实验。
- 禁止只有 notebook 截图而无可复现脚本。
- 所有训练必须支持 seed。
- 所有环境必须明确 observation/action 的 shape、unit、range。
- 所有 reward term 必须命名并单独记录。
- 所有 action 必须有合理的 clipping / scaling。

## 4. 不允许 RL 直接掩盖传统控制问题
失败必须先区分：ENVIRONMENT / CONTROL / RL / EVALUATION。
禁止通过修改 reward 去掩盖明显的物理/控制 bug。

## 5. Residual RL
必须显式记录：
`u_total = u_base + alpha * delta_u_rl`
并分别记录 `u_base`、`delta_u_rl`、`u_total`；必须提供 `alpha=0` classical baseline。

## 6. 公平比较
PPO vs SAC、classical vs RL、RL vs residual 必须保持同一环境、reset 分布、评估 seed，并记录训练预算。

## 7. 结果记录
每个 run：`results/<date>_<task>_<run-id>/`
至少保存 git commit、seed、config、training steps、wall time、mean return、success rate、failure counts、checkpoint path。

## 8. Anti-loop
同一根因最多：一次诊断 + 一次修复 + 一次备选。仍失败则停止并输出工程升级报告。

## 9. 研究顺序
`基础 → Pendulum → FR3 Reach → Cube Push → Residual → Force-aware → Safety/QP → Randomization → Dual-arm`。

## 10. Post-task report
```text
=== POST-TASK REPORT ===
Task:
Status: PASS / PARTIAL / BLOCKED
Learning objective achieved: yes/no
Engineering objective achieved: yes/no
What was learned:
Implemented:
Experiments:
Key metrics:
Failure analysis:
Files changed:
Recommended next task:
Git branch / commit:
```
