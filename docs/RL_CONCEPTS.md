# RL Concepts — 只保留机器人 RL 必需概念

## MDP
机器人 RL 可抽象为：`(state, action, transition, reward, discount)`。

## State / Observation
policy 往往拿到 observation，而不是完整真实 state。机器人中常见 observation：`q, qdot, ee pose, target, object pose, wrench, previous action`。

## Action
本仓库按风险从低到高学习：
1. controller parameter；
2. Cartesian `Δx / Δv`；
3. joint velocity；
4. joint torque（后期）。

## Reward
Reward 是“训练信号”，不是 benchmark 指标。成功率必须独立评估。

## V / Q
- `V(s)`：从状态 s 出发，按当前策略继续走，未来累计回报的期望。
- `Q(s,a)`：在状态 s 先做动作 a，再按当前策略继续走，未来累计回报的期望。

## Policy
`π(a|s)`：给定 observation/state 后如何选择动作。

## Actor-Critic
Actor 负责出动作，Critic 负责评估动作/状态质量。

## PPO
重点理解：on-policy、advantage、clipped objective、稳定更新。

## SAC
重点理解：off-policy、双 Q、entropy、sample efficiency。

## Residual RL
`u_total = u_base + Δu_RL`。传统控制器提供基础稳定性，RL 学剩余修正。
