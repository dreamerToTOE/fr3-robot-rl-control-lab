# RL05 对照结果 — Random / Classical DLS / SAC / PPO（FR3 Cartesian Reach）

评估协议：同一组 100 个**已验证可达**目标；关节精确复位到 READY；30 Hz 控制；
成功判据 `|ee - target| <= 0.02 m` 连续保持 3 个控制步；上限 100 控制步；评估期不更新参数。

| Policy | Success | Mean final err (m) | Median err (m) | Time-to-success (s) | Mean return | Ep len | Sat ratio | Smoothness RMS | Limit viol | Collision steps |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.0% | 0.2152 | 0.2023 | — | -4.97 | 100.0 | 47.7% | 1.4094 | 0 | 0 |
| Classical DLS | 90.0% | 0.0136 | 0.0113 | 1.461 | 9.00 | 49.5 | 0.0% | 0.0033 | 0 | 0 |
| Oracle (myopic) | 70.0% | 0.0326 | 0.0148 | 1.929 | 6.46 | 70.5 | 50.2% | 0.1301 | 0 | 0 |
| SAC | 12.0% | 0.0863 | 0.0611 | 1.786 | 0.03 | 94.4 | 54.3% | 0.2868 | 0 | 0 |
| PPO | 12.0% | 0.0901 | 0.0859 | 1.592 | -0.90 | 93.7 | 97.3% | 0.2156 | 0 | 0 |
