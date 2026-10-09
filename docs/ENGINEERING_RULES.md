# Engineering Rules

## 推荐 action 接口顺序
`controller params → Cartesian delta → Cartesian velocity → joint velocity → torque`

## 初期不要做
- 14DoF 双臂直接 torque RL；
- 复杂视觉输入 + 接触控制一起上；
- reward term 超过必要数量；
- 环境还不稳定就盲调 PPO；
- 只看 reward 不看机器人行为。

## 训练 debug 顺序
1. reset 是否正确；
2. observation 是否 finite / scale 合理；
3. action 方向和单位是否正确；
4. random policy 是否会触发合理状态变化；
5. classical baseline 是否可运行；
6. reward 单项是否符合预期；
7. 最后才调 RL hyperparameters。
