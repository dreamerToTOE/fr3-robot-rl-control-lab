# RL10 — Force-aware RL

## Goal
让 policy 使用末端 wrench/contact 特征。

## Engineering
加入 filtered wrench、force history 或 compact contact state；防止高频噪声直接进入 policy。

## PASS
证明 force observation 对 jam/force peak/成功率中的至少一个指标有可测影响。
