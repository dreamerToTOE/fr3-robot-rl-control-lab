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
