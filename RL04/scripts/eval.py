"""RL04 —— 独立评估脚本（训练/评估严格分离）。

用法:
    # 最终模型评估（20 episodes, 固定种子 1000..1019）
    python scripts/eval.py --algo ppo
    python scripts/eval.py --algo sac
    python scripts/eval.py --algo random

    # 逐 checkpoint 评估，生成 eval_return vs steps 曲线数据
    python scripts/eval.py --algo ppo --mode sweep
    python scripts/eval.py --algo sac --mode sweep

评估协议（PPO/SAC/Random 完全一致）:
    - 环境 Pendulum-v1, 每个 episode 用固定 seed = eval_seed_base + i
    - deterministic action（Random 用 env.action_space.sample()）
    - 每个 episode 固定 200 步 TimeLimit；truncation 不视为失败
    - 评估过程中不更新任何网络参数

指标容差（显式记录）:
    upright: |theta| <= 0.2 rad 且 |theta_dot| <= 1.0 rad/s
    saturation: |u| >= 0.99 * 2.0 N·m
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import rl04_common as C  # noqa: E402


def run_episode(env, policy_fn, seed: int, episode_idx: int) -> C.EpisodeRecord:
    obs, _ = env.reset(seed=seed)
    theta_trace, theta_dot_trace, torque_trace, ret_trace = [], [], [], []
    ep_return = 0.0
    terminated = truncated = False
    step = 0
    while True:
        action = np.asarray(policy_fn(obs), dtype=np.float64).reshape(-1)
        obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        theta = C.obs_to_theta(obs)
        tau = float(action[0])
        ep_return += float(reward)
        theta_trace.append(theta)
        theta_dot_trace.append(float(obs[2]))
        torque_trace.append(tau)
        ret_trace.append(ep_return)
        step += 1
        if terminated or truncated:
            break

    th = np.array(theta_trace)
    thd = np.array(theta_dot_trace)
    tq = np.array(torque_trace)
    upright_mask = (np.abs(th) <= C.UPRIGHT_ANGLE_TOL) & (np.abs(thd) <= C.UPRIGHT_ANGULAR_VEL_TOL)
    return C.EpisodeRecord(
        episode=episode_idx,
        seed=seed,
        return_=ep_return,
        length=step,
        mean_abs_theta=float(np.abs(th).mean()),
        mean_abs_theta_dot=float(np.abs(thd).mean()),
        mean_abs_torque=float(np.abs(tq).mean()),
        max_abs_torque=float(np.abs(tq).max()),
        saturation_ratio=float((np.abs(tq) >= C.SATURATION_RATIO_THRESHOLD * C.MAX_TORQUE).mean()),
        upright_fraction=float(upright_mask.mean()),
        terminated=bool(terminated),
        truncated=bool(truncated),
        theta_trace=theta_trace,
        theta_dot_trace=theta_dot_trace,
        torque_trace=torque_trace,
        return_trace=ret_trace,
    )


def write_episodes_csv(path: Path, records: list[C.EpisodeRecord]) -> None:
    fields = [
        "episode", "seed", "return", "length", "mean_abs_theta_rad", "mean_abs_theta_dot_rad_s",
        "mean_abs_torque_nm", "max_abs_torque_nm", "saturation_ratio", "upright_fraction",
        "terminated", "truncated",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in records:
            w.writerow([
                r.episode, r.seed, round(r.return_, 6), r.length,
                round(r.mean_abs_theta, 6), round(r.mean_abs_theta_dot, 6),
                round(r.mean_abs_torque, 6), round(r.max_abs_torque, 6),
                round(r.saturation_ratio, 6), round(r.upright_fraction, 6),
                r.terminated, r.truncated,
            ])


def write_trace_csv(path: Path, records: list[C.EpisodeRecord]) -> None:
    rows = C.records_to_trace_rows(records)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def make_policy(algo: str, model, env):
    if algo == "random":
        return lambda obs: env.action_space.sample()
    return lambda obs: model.predict(obs, deterministic=True)[0]


def load_model(algo: str, model_path: Path):
    from stable_baselines3 import PPO, SAC

    cls = PPO if algo == "ppo" else SAC
    return cls.load(str(model_path), device="cpu")


def eval_final(algo: str, run_dir: Path, episodes: int, seed_base: int, model_path: Path | None):
    env = C.make_env(C.ENV_ID)
    model = None
    if algo != "random":
        model = load_model(algo, model_path or (run_dir / "final_model.zip"))
    policy = make_policy(algo, model, env)

    records = []
    for i in range(episodes):
        seed = seed_base + i
        records.append(run_episode(env, policy, seed, i))
        if (i + 1) % 5 == 0:
            print(f"  [{algo}] episode {i + 1}/{episodes} done")
    env.close()

    meta = {
        "policy": "%s deterministic" % algo if algo != "random" else "random uniform",
        "model_path": str(model_path or (run_dir / "final_model.zip")) if algo != "random" else None,
        "deterministic": algo != "random",
        "eval_seeds": [seed_base, seed_base + episodes - 1],
        "episode_time_limit": 200,
    }
    metrics = C.summarise_episodes(records, meta)
    C.dump_json(run_dir / "eval_metrics.json", metrics)
    write_episodes_csv(run_dir / "eval_episodes.csv", records)
    write_trace_csv(run_dir / "eval_trace.csv", records)

    print(
        f"[eval:{algo}] mean_return={metrics['mean_episode_return']:.2f} "
        f"± {metrics['std_episode_return']:.2f} | upright={metrics['upright_fraction']:.2%} "
        f"| mean|theta|={metrics['mean_abs_angle_error_rad']:.3f} rad "
        f"| sat={metrics['action_saturation_ratio']:.2%}"
    )
    return metrics


def eval_sweep(algo: str, run_dir: Path, episodes: int, seed_base: int) -> None:
    """对 checkpoints/ 下所有 *.zip（按步数排序）+ final_model 做评估。"""
    env = C.make_env(C.ENV_ID)
    ckpt_dir = run_dir / "checkpoints"
    items = []
    for p in sorted(ckpt_dir.glob("*.zip")):
        m = re.search(r"_(\d+)_steps$", p.stem)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    final = run_dir / "final_model.zip"
    if final.exists():
        steps = int(C.read_json(run_dir / "metadata.json")["total_timesteps_actual"])
        items.append((steps, final))
    items.sort(key=lambda x: x[0])

    rows = []
    for steps, path in items:
        model = load_model(algo, path)
        policy = make_policy(algo, model, env)
        recs = [run_episode(env, policy, seed_base + i, i) for i in range(episodes)]
        summ = C.summarise_episodes(recs)
        rows.append(
            {
                "algo": algo,
                "env_step": steps,
                "model": path.name,
                "mean_return": round(summ["mean_episode_return"], 4),
                "std_return": round(summ["std_episode_return"], 4),
                "upright_fraction": round(summ["upright_fraction"], 4),
                "mean_abs_theta_rad": round(summ["mean_abs_angle_error_rad"], 4),
                "action_saturation_ratio": round(summ["action_saturation_ratio"], 4),
            }
        )
        print(f"  [sweep:{algo}] step={steps:6d} mean_return={summ['mean_episode_return']:.2f}")
    env.close()

    with open(run_dir / "eval_vs_steps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep:{algo}] -> {run_dir / 'eval_vs_steps.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "sac", "random"])
    ap.add_argument("--mode", default="final", choices=["final", "sweep"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--run-id", default=None, help="默认 <algo>_seed42")
    ap.add_argument("--model", default=None, help="显式指定模型路径（final 模式）")
    args = ap.parse_args()

    seed = 42
    run_id = args.run_id or f"{args.algo}_seed{seed}"
    run_dir = C.ensure_dir(C.RESULTS_ROOT / run_id)

    if args.mode == "sweep":
        eval_sweep(args.algo, run_dir, args.episodes, args.seed_base)
    else:
        eval_final(args.algo, run_dir, args.episodes, args.seed_base,
                   Path(args.model) if args.model else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
