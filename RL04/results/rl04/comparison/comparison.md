# RL04 对照结果 — Random vs PPO vs SAC (Pendulum-v1)

评估协议：每个策略 20 个独立 episode，固定 seed = 1000..1019，deterministic 动作，每 episode 固定 200 步 TimeLimit（truncation 不计为失败）。

容差定义：upright = |theta| ≤ 0.2 rad 且 |theta_dot| ≤ 1.0 rad/s；saturation = |u| ≥ 0.99 × 2.0 N·m。

| Policy | Mean Return | Std Return | Mean |theta| (rad) | Mean |theta_dot| (rad/s) | Mean |u| (N·m) | Max |u| (N·m) | Saturation | Upright frac |
|---|---|---|---|---|---|---|---|---|
| RANDOM UNIFORM | -1258.78 | 269.56 | 2.108 | 2.915 | 0.997 | 2.000 | 1.23% | 0.05% |
| PPO DETERMINISTIC | -1101.13 | 199.94 | 1.769 | 3.454 | 0.275 | 0.781 | 0.00% | 0.75% |
| SAC DETERMINISTIC | -151.20 | 81.53 | 0.338 | 0.497 | 0.671 | 1.999 | 3.50% | 84.00% |
