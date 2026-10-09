# RL09 — Residual RL

## Goal
让经典控制器负责基础动作，RL 只学习修正量。

## Core
`u_total = u_base + alpha * delta_u_rl`

## Suggested Base
Admittance / Cartesian push controller。

## Suggested Residual Action
ΔF_target、Δy、Δyaw、ΔB 中先选 1–2 个，不要一次全开。

## PASS
classical (`alpha=0`) 与 residual 在同 benchmark 下有可解释对比。
