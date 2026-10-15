"""RL04 —— PPO / SAC 训练脚本（Stable-Baselines3）。

用法:
    python scripts/train.py --algo ppo --config configs/ppo_seed42.yaml
    python scripts/train.py --algo sac --config configs/sac_seed42.yaml

产物（RESULTS_ROOT/ppo_seed42/ 等）:
    config.json         实验配置（实际生效值）
    metadata.json       git commit / 版本 / seed / wall time / checkpoint 列表
    final_model.zip     最终模型
    checkpoints/        周期 checkpoint
    train_metrics.csv   按 environment step 记录的 episode return
    logs/progress.csv   SB3 自身训练日志
    logs/stdout.log     训练 stdout
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import rl04_common as C  # noqa: E402


def _rel_to_root(p) -> str:
    """把路径转成相对 RL04 根目录的字符串，越界时回退为绝对路径。"""
    try:
        return str(Path(p).resolve().relative_to(C.RL04_ROOT))
    except Exception:
        return str(p)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def build_model(algo: str, env, cfg: dict):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.logger import configure

    policy_kwargs = dict(net_arch=list(cfg["net_arch"]))
    common = dict(
        policy=cfg["policy"],
        env=env,
        learning_rate=float(cfg["learning_rate"]),
        gamma=float(cfg["gamma"]),
        policy_kwargs=policy_kwargs,
        seed=int(cfg["seed"]),
        verbose=1,
        tensorboard_log=None,
    )
    if algo == "ppo":
        model = PPO(
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            n_epochs=int(cfg["n_epochs"]),
            clip_range=float(cfg["clip_range"]),
            gae_lambda=float(cfg["gae_lambda"]),
            ent_coef=float(cfg["ent_coef"]),
            vf_coef=float(cfg["vf_coef"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            **common,
        )
    elif algo == "sac":
        model = SAC(
            batch_size=int(cfg["batch_size"]),
            buffer_size=int(cfg["buffer_size"]),
            tau=float(cfg["tau"]),
            train_freq=int(cfg["train_freq"]),
            gradient_steps=int(cfg["gradient_steps"]),
            learning_starts=int(cfg["learning_starts"]),
            ent_coef=cfg["ent_coef"],
            **common,
        )
    else:
        raise ValueError(f"unknown algo: {algo}")
    return model, configure


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "sac"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None, help="默认 <algo>_seed<seed>")
    ap.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    ap.add_argument("--total-timesteps", type=int, default=None, help="覆盖配置中的训练步数")
    args = ap.parse_args()

    cfg = C.load_yaml_config(Path(args.config))
    algo = args.algo
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg["seed"])
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps

    run_id = args.run_id or f"{algo}_seed{seed}"
    run_dir = C.ensure_dir(C.RESULTS_ROOT / run_id)
    ckpt_dir = C.ensure_dir(run_dir / "checkpoints")
    logs_dir = C.ensure_dir(run_dir / "logs")

    # 导入 SB3（延迟，便于先跑 smoke test）
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    set_global_seed(seed)
    train_env = Monitor(C.make_env(cfg["env_id"], seed=seed))

    model, configure = build_model(algo, train_env, cfg)
    logger = configure(str(logs_dir), ["csv", "stdout"])
    model.set_logger(logger)

    callbacks = []
    if C.EpisodeMetricsCallback is not None:
        ep_cb = C.EpisodeMetricsCallback()
        callbacks.append(ep_cb)
    callbacks.append(
        CheckpointCallback(
            save_freq=int(cfg["checkpoint_freq"]),
            save_path=str(ckpt_dir),
            name_prefix=f"{algo}_seed{seed}",
            verbose=1,
        )
    )

    # 记录实际生效配置
    config_record = dict(cfg)
    config_record.update(
        {
            "algo": algo,
            "run_id": run_id,
            "policy_kwargs": {"net_arch": list(cfg["net_arch"])},
            "observation_space": str(train_env.observation_space),
            "action_space": str(train_env.action_space),
            "action_low": np.asarray(train_env.action_space.low).tolist(),
            "action_high": np.asarray(train_env.action_space.high).tolist(),
        }
    )
    try:
        config_record["effective_hyperparams"] = {
            k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
            for k, v in model.get_params().items()
            if k not in ("policy", "env")
        }
    except Exception as exc:
        config_record["effective_hyperparams_error"] = str(exc)
    C.dump_json(run_dir / "config.json", config_record)

    print(f"[rl04] training {algo} seed={seed} steps={cfg['total_timesteps']} -> {run_dir}")
    t0 = time.perf_counter()
    model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callbacks, progress_bar=False)
    wall = time.perf_counter() - t0

    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"[rl04] training done in {wall:.1f}s -> {final_path}")

    # 写 train_metrics.csv
    rows = ep_cb.rows if C.EpisodeMetricsCallback is not None else []
    tm_path = run_dir / "train_metrics.csv"
    with open(tm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["env_step", "episode_return", "episode_length"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("env_step", "episode_return", "episode_length")})

    # metadata.json
    versions = C.collect_env_versions()
    ckpts = sorted(p.name for p in ckpt_dir.glob("*.zip"))
    meta = {
        "algo": algo,
        "run_id": run_id,
        "seed": seed,
        "env_id": cfg["env_id"],
        "total_timesteps_requested": int(cfg["total_timesteps"]),
        "total_timesteps_actual": int(model.num_timesteps),
        "wall_time_s": round(wall, 3),
        "num_episodes_logged": len(rows),
        "final_model": str(final_path.relative_to(C.RL04_ROOT)),
        "checkpoints": ckpts,
        "versions": versions,
        "git_commit_repo_ref": C.git_commit_of(C.REF_REPO),
        "git_commit_local": C.git_commit_of(C.RL04_ROOT.parent),
        "config_path": _rel_to_root(args.config),
        "host": C.os_env_summary(),
    }
    C.dump_json(run_dir / "metadata.json", meta)
    print(f"[rl04] wrote {tm_path.name}, metadata.json, {len(ckpts)} checkpoints")
    train_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
