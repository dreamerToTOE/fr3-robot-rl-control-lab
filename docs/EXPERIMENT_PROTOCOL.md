# Experiment Protocol

## 每个训练 run 必须固定/记录
- task / environment version
- git commit
- algorithm
- seed
- total environment steps
- num envs
- physics dt / control decimation
- observation terms and scaling
- action terms and scaling
- reward terms and weights
- termination rules
- optimizer / learning rate / batch size
- network architecture

## 评估与训练分离
训练期间的 episode return 不能作为最终 benchmark。评估必须使用固定 seed 集和独立 deterministic/stochastic policy 设置。

## 最低评估指标
Reach：success rate、final error、time-to-success、action smoothness。
Push/Insertion：success rate、time、contact force RMS/max、jam rate、lateral/yaw error。
Dual-arm：再加 relative-pose error、internal wrench、min distance。

## 图表
每个正式实验至少输出：
1. return vs steps；
2. success rate vs steps；
3. episode length；
4. action magnitude / saturation；
5. task-specific error/force plot。
