"""RL04 —— 环境 Smoke Test。

在进入训练之前验证 Pendulum-v1 接口完全符合预期：
1. 能创建 / reset
2. observation 维度 = (3,)，且为 [cos, sin, theta_dot]
3. action space 为 Box(1,)，范围 [-2, 2]
4. step() 返回 5 元组且类型合法
5. reward / observation 无 NaN / Inf
6. 能完整运行一个 episode（含 TimeLimit truncation 语义）

输出: results/rl04/smoke_test.json
退出码非 0 表示环境不合格，禁止进入训练。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import rl04_common as C  # noqa: E402


def main() -> int:
    checks: dict[str, object] = {}
    ok = True

    env = C.make_env(seed=42)
    checks["env_id"] = C.ENV_ID

    # 1. action / observation space
    obs_space = env.observation_space
    act_space = env.action_space
    checks["observation_space"] = str(obs_space)
    checks["observation_shape"] = list(obs_space.shape)
    checks["action_space"] = str(act_space)
    checks["action_low"] = np.asarray(act_space.low).tolist()
    checks["action_high"] = np.asarray(act_space.high).tolist()

    if tuple(obs_space.shape) != (3,):
        ok = False
        checks["fail_obs_shape"] = "expected (3,)"
    if tuple(act_space.shape) != (1,):
        ok = False
        checks["fail_act_shape"] = "expected (1,)"
    if not np.allclose(act_space.low, [-C.MAX_TORQUE]) or not np.allclose(act_space.high, [C.MAX_TORQUE]):
        ok = False
        checks["fail_act_range"] = "expected [-2, 2]"

    # 2. reset
    obs, info = env.reset(seed=42)
    obs = np.asarray(obs, dtype=np.float64)
    checks["reset_obs"] = obs.tolist()
    checks["reset_info_keys"] = sorted(info.keys()) if isinstance(info, dict) else "n/a"
    if obs.shape != (3,):
        ok = False
        checks["fail_reset_shape"] = list(obs.shape)
    if not np.all(np.isfinite(obs)):
        ok = False
        checks["fail_reset_nonfinite"] = True
    # cos^2 + sin^2 == 1
    norm = float(obs[0] ** 2 + obs[1] ** 2)
    checks["reset_cos2sin2"] = norm
    if not (0.99 <= norm <= 1.01):
        ok = False
        checks["fail_cos2sin2"] = norm

    # 3. step() 合法性（采样若干动作）
    rng = np.random.default_rng(0)
    step_probe = []
    for _ in range(5):
        a = rng.uniform(act_space.low, act_space.high).astype(np.float32)
        out = env.step(a)
        if len(out) != 5:
            ok = False
            checks["fail_step_len"] = len(out)
            break
        o2, r, terminated, truncated, info2 = out
        o2 = np.asarray(o2, dtype=np.float64)
        finite = bool(np.all(np.isfinite(o2)) and np.isfinite(r))
        if not finite:
            ok = False
        step_probe.append(
            {
                "action": np.asarray(a).tolist(),
                "obs": o2.tolist(),
                "reward": float(r),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "finite": finite,
            }
        )
    checks["step_probe"] = step_probe

    # 4. 完整 episode（用零动作，检查 TimeLimit truncation）
    obs, _ = env.reset(seed=123)
    ep_return = 0.0
    steps = 0
    max_abs_torque_seen = 0.0
    while True:
        action = np.array([0.0], dtype=np.float32)
        obs, r, terminated, truncated, _ = env.step(action)
        if not (np.isfinite(r) and np.all(np.isfinite(obs))):
            ok = False
            checks["fail_episode_nonfinite"] = True
            break
        ep_return += float(r)
        steps += 1
        max_abs_torque_seen = max(max_abs_torque_seen, abs(float(action[0])))
        if terminated or truncated:
            break
    checks["episode_steps"] = steps
    checks["episode_return"] = ep_return
    checks["episode_truncated_at_timelimit"] = bool(truncated)
    checks["episode_terminated"] = bool(terminated)
    if steps != 200:
        ok = False
        checks["fail_episode_length"] = steps
    if not truncated:
        ok = False
        checks["fail_no_truncation"] = "expected TimeLimit truncation at 200 steps"

    # 5. 一个随机策略 episode，检查随机动作也不会发散
    obs, _ = env.reset(seed=7)
    rand_return = 0.0
    rand_ok = True
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(a)
        rand_return += float(r)
        if not (np.isfinite(r) and np.all(np.isfinite(obs))):
            rand_ok = False
            break
        if terminated or truncated:
            break
    checks["random_policy_return"] = rand_return
    checks["random_policy_finite"] = rand_ok
    if not rand_ok:
        ok = False

    env.close()

    checks["overall"] = "PASS" if ok else "FAIL"
    checks["versions"] = C.collect_env_versions()

    C.dump_json(C.RESULTS_ROOT / "smoke_test.json", checks)

    print("=" * 60)
    print("RL04 SMOKE TEST:", checks["overall"])
    print("  obs_space   :", checks["observation_space"])
    print("  act_space   :", checks["action_space"], checks["action_low"], checks["action_high"])
    print("  episode     :", steps, "steps, truncated:", truncated, "return:", round(ep_return, 2))
    print("  random ret  :", round(rand_return, 2))
    print("  -> results/rl04/smoke_test.json")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
