#!/usr/bin/env python3
"""RL05 —— SAC / PPO 训练（Isaac Sim 物理 + Stable-Baselines3）。

必须在 Isaac Sim 自带 Python 下运行，并把 SB3 通过 PYTHONPATH 注入：
    cd /home/ubuntu2004/isaacsim-4.5.0
    PYTHONPATH=<RL05>/.pydeps ./python.sh <RL05>/scripts/train.py --algo sac --config <cfg>

产物（RL05/results/rl05/<run-id>/）:
    config.json  metadata.json  final_model.zip  vecnormalize.pkl
    checkpoints/ train_metrics.csv  episodes.csv  logs/

注意：所有训练都在同一进程内完成（SimulationApp 每进程只能创建一个）。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RL05 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL05 / "envs"))
sys.path.insert(0, str(RL05 / "controllers"))

RESULTS = RL05 / "results" / "rl05"


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(RL05))
    except Exception:
        return str(p)


def _jsonable(v):
    """把 torch Tensor / numpy 标量等转成可 JSON 序列化的值。"""
    try:
        import numpy as _np
        import torch as _torch
        if isinstance(v, _torch.Tensor):
            return v.detach().cpu().tolist()
        if isinstance(v, _np.ndarray):
            return v.tolist()
        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    if isinstance(v, (int, float, str, bool, list, dict, type(None))):
        return v
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["sac", "ppo"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    algo = args.algo
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = int(args.total_timesteps)
    seed = int(cfg["seed"])
    run_id = args.run_id or f"{algo}_seed{seed}"
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"utc": datetime.now(timezone.utc).isoformat(), "algo": algo,
                    "run_id": run_id, "success": False}
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})

        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.logger import configure

        from fr3_reach_env import FR3ReachEnv, FR3ReachSim, ReachConfig

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ---- 环境 ----
        env_cfg = ReachConfig(
            physics_dt=cfg.get("physics_dt", 1.0 / 120.0),
            decimation=int(cfg.get("decimation", 4)),
            max_control_steps=int(cfg.get("max_control_steps", 100)),
            action_scale=float(cfg.get("action_scale", 0.01)),
            success_tol=float(cfg.get("success_tol", 0.02)),
            hold_steps=int(cfg.get("hold_steps", 3)),
            reset_joint_noise=float(cfg.get("reset_joint_noise", 0.05)),
            progress_coef=float(cfg.get("progress_coef", 1.0)),
            success_bonus=float(cfg.get("success_bonus", 10.0)),
            action_penalty_coef=float(cfg.get("action_penalty_coef", 0.01)),
            smoothness_coef=float(cfg.get("smoothness_coef", 0.02)),
        )
        sim = FR3ReachSim(cfg=env_cfg)
        pool = np.asarray(json.loads((RL05 / "assets" / "targets_train.json").read_text())["targets"],
                          dtype=np.float64)

        def make_one():
            e = FR3ReachEnv(sim, env_cfg, target_pool=pool, seed=seed, record_trace=False)
            return Monitor(e, info_keywords=("is_success",))

        vec = DummyVecEnv([make_one])
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

        # ---- 回调：按 env step 记录 episode 结果 ----
        class EpisodeLog(BaseCallback):
            def __init__(self):
                super().__init__(0)
                self.rows: list[dict] = []
                self.successes = 0
                self.episodes = 0

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    ep = info.get("episode")
                    if ep is not None:
                        ok = bool(info.get("is_success", False))
                        self.episodes += 1
                        self.successes += int(ok)
                        self.rows.append({
                            "env_step": int(self.num_timesteps),
                            "episode_return": round(float(ep["r"]), 6),
                            "episode_length": int(ep["l"]),
                            "is_success": int(ok),
                            "final_distance": round(float(info.get("distance", float("nan"))), 6),
                        })
                return True

        ep_log = EpisodeLog()
        callbacks = [ep_log]
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if cfg.get("checkpoint_freq"):
            callbacks.append(CheckpointCallback(save_freq=int(cfg["checkpoint_freq"]),
                                                save_path=str(ckpt_dir),
                                                name_prefix=f"{algo}_seed{seed}", verbose=0))

        # ---- 模型 ----
        net = list(cfg.get("net_arch", [256, 256]))
        common = dict(policy=cfg.get("policy", "MlpPolicy"), env=vec,
                      learning_rate=float(cfg["learning_rate"]), gamma=float(cfg["gamma"]),
                      policy_kwargs=dict(net_arch=net), seed=seed, verbose=1,
                      tensorboard_log=None)
        if algo == "sac":
            model = SAC(batch_size=int(cfg.get("batch_size", 256)),
                        buffer_size=int(cfg.get("buffer_size", 300000)),
                        tau=float(cfg.get("tau", 0.005)),
                        train_freq=int(cfg.get("train_freq", 1)),
                        gradient_steps=int(cfg.get("gradient_steps", 1)),
                        learning_starts=int(cfg.get("learning_starts", 500)),
                        ent_coef=cfg.get("ent_coef", "auto"), **common)
        else:
            model = PPO(n_steps=int(cfg.get("n_steps", 1024)),
                        batch_size=int(cfg.get("batch_size", 256)),
                        n_epochs=int(cfg.get("n_epochs", 10)),
                        clip_range=float(cfg.get("clip_range", 0.2)),
                        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
                        ent_coef=float(cfg.get("ent_coef", 0.0)),
                        vf_coef=float(cfg.get("vf_coef", 0.5)),
                        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)), **common)

        logs = run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(logs), ["csv", "stdout"]))

        (run_dir / "config.json").write_text(json.dumps({
            **cfg, "algo": algo, "run_id": run_id,
            "env": {"physics_dt": env_cfg.physics_dt, "decimation": env_cfg.decimation,
                    "control_hz": env_cfg.control_hz, "max_control_steps": env_cfg.max_control_steps,
                    "action_scale": env_cfg.action_scale, "success_tol": env_cfg.success_tol,
                    "hold_steps": env_cfg.hold_steps},
            "observation_dim": 20, "action_dim": 3,
            "observation": "[q(7), qdot(7), ee_error(3), prev_action(3)]",
            "ee_index": sim.ee_idx, "link_names_head": sim.link_names[:24],
            "target_pool_size": int(pool.shape[0]),
        }, indent=2), encoding="utf-8")

        print(f"[rl05] training {algo} seed={seed} steps={cfg['total_timesteps']}", flush=True)
        t0 = time.perf_counter()
        model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callbacks, progress_bar=False)
        wall = time.perf_counter() - t0

        model.save(str(run_dir / "final_model.zip"))
        vec.save(str(run_dir / "vecnormalize.pkl"))
        # 额外导出可读的归一化统计量，评估脚本无需依赖 pkl
        try:
            (run_dir / "vecnormalize_stats.json").write_text(json.dumps({
                "obs_mean": np.asarray(vec.obs_rms.mean, dtype=float).tolist(),
                "obs_var": np.asarray(vec.obs_rms.var, dtype=float).tolist(),
                "clip_obs": float(vec.clip_obs),
                "epsilon": float(vec.epsilon),
                "norm_obs": bool(vec.norm_obs),
                "norm_reward": bool(vec.norm_reward),
            }, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[rl05] WARN could not dump vecnormalize stats: {exc}", flush=True)
        vec.close()

        with open(run_dir / "train_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["env_step", "episode_return", "episode_length",
                                              "is_success", "final_distance"])
            w.writeheader()
            w.writerows(ep_log.rows)

        versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                    "cuda": bool(torch.cuda.is_available()),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "gymnasium": gym.__version__,
                    "stable_baselines3": __import__("stable_baselines3").__version__,
                    "numpy": np.__version__}
        # 记录实际生效的超参（SB3 各版本 API 不一致，做多级回退）
        hyper: dict = {}
        for attr in ("get_params", "get_parameters"):
            fn = getattr(model, attr, None)
            if callable(fn):
                try:
                    hyper = {k: v for k, v in fn().items() if k not in ("policy", "env")}
                    break
                except Exception:
                    continue
        if not hyper:
            hyper = {k: cfg[k] for k in ("learning_rate", "gamma", "batch_size", "buffer_size",
                                         "tau", "n_steps", "n_epochs", "clip_range", "gae_lambda",
                                         "net_arch", "ent_coef", "learning_starts", "train_freq",
                                         "gradient_steps") if k in cfg}
            hyper["_source"] = "config (get_params unavailable in this SB3 version)"
        hyper = {k: _jsonable(v) for k, v in hyper.items()}
        (run_dir / "metadata.json").write_text(json.dumps({
            "algo": algo, "run_id": run_id, "seed": seed,
            "total_timesteps_requested": int(cfg["total_timesteps"]),
            "total_timesteps_actual": int(model.num_timesteps),
            "wall_time_s": round(wall, 2),
            "env_steps_per_s": round(model.num_timesteps / max(wall, 1e-9), 2),
            "episodes": ep_log.episodes, "episode_successes": ep_log.successes,
            "train_success_rate": (ep_log.successes / ep_log.episodes) if ep_log.episodes else None,
            "sim": {"physics_dt": env_cfg.physics_dt, "decimation": env_cfg.decimation,
                    "control_hz": env_cfg.control_hz, "num_envs": 1},
            "versions": versions, "hyperparams": hyper,
            "config_path": rel(args.config), "final_model": rel(run_dir / "final_model.zip"),
            "checkpoints": sorted(p.name for p in ckpt_dir.glob("*.zip")),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        report.update({"success": True, "wall_time_s": round(wall, 2),
                       "env_steps_per_s": round(model.num_timesteps / max(wall, 1e-9), 2),
                       "episodes": ep_log.episodes, "episode_successes": ep_log.successes})
        (run_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"TRAIN_OK {algo} seed={seed} steps={model.num_timesteps} wall={wall:.1f}s "
              f"eps/s={report['env_steps_per_s']} episodes={ep_log.episodes} "
              f"train_success={report['episode_successes']}", flush=True)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"TRAIN_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
