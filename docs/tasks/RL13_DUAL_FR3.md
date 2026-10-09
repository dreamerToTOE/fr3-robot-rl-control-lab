# RL13 — Dual-FR3 High-level RL

## Goal
从单臂迁移到双臂，但不直接做 14DoF torque policy。

## Action
优先 object-level / relative-pose / Cartesian corrections / internal-force target。

## PASS
双臂 policy 能在传统低层控制器上完成一个可复现协调任务。
