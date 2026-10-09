# RL11 — Safety Filter / QP

## Goal
将 RL action 通过约束层再执行。

## Constraints
joint position/velocity、action bounds、collision distance；后续可加 torque/contact force。

## PASS
保存 raw action 与 filtered action，并证明安全层不会悄悄改变 benchmark 定义。
