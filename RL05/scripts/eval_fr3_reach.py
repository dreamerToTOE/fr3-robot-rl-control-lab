#!/usr/bin/env python3
"""RL05 —— 统一独立评估：Random / Classical(DLS) / SAC / PPO。

评估协议（四种策略完全一致）：
    目标集   : assets/targets_eval.json 中固定的 100 个已验证可达目标
    初始状态 : 关节精确复位到 READY（reset_joint_noise = 0），保证可比
    控制频率 : 30 Hz（physics_dt=1/120, decimation=4）
    成功判据 : |ee - target| <= 0.02 m 连续保持 3 个控制步
    时间上限 : 100 个控制步（3.33 s 仿真时间）
    网络更新 : 评估期间禁止更新参数（.predict 只做前向推理）

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    PYTHONPATH=<RL05>/.pydeps ./python.sh <RL05>/scripts/eval_fr3_reach.py --policy classical
    PYTHONPATH=<RL05>/.pydeps ./python.sh <RL05>/scripts/eval_fr3_reach.py --policy random
    PYTHONPATH=<RL05>/.pydeps ./python.sh <RL05>/scripts/eval_fr3_reach.py --policy sac --run-id sac_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
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
READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["classical", "random", "sac", "ppo", "oracle"])
    ap.add_argument("--run-id", default=None, help="RL 模型所在 run（默认 <policy>_seed42）")
    ap.add_argument("--tag", default=None, help="输出子目录名（默认 <policy>_eval）")
    ap.add_argument("--model", default=None, help="显式指定模型 zip（用于逐 checkpoint 评估）")
    ap.add_argument("--max-episodes", type=int, default=None, help="只用前 N 个评估目标")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    pol = args.policy
    tag = args.tag or f"{pol}_eval"
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rep: dict = {"utc": datetime.now(timezone.utc).isoformat(), "policy": pol, "success": False}
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})

        from fr3_reach_env import FR3ReachEnv, FR3ReachSim, ReachConfig, Q_MIN, Q_MAX
        from dls_ik import default_fr3_controller
        from classic_reach import contact_force_norm, contact_api_available, reach_rollout

        eval_path = RL05 / "assets" / "targets_eval.json"
        meta = json.loads(eval_path.read_text())
        targets = np.asarray(meta["targets"], dtype=np.float64)
        if args.max_episodes:
            targets = targets[: args.max_episodes]
        rep["n_targets"] = int(targets.shape[0])
        rep["success_tol"] = meta["success_tol"]
        rep["hold_steps"] = meta["hold_steps"]

        cfg = ReachConfig(reset_joint_noise=0.0)
        sim = FR3ReachSim(cfg=cfg)
        ctrl = default_fr3_controller(cfg.control_dt)
        a_scale = cfg.action_scale
        rep["contact_api_available"] = bool(contact_api_available(sim))

        model = None
        norm_stats = None
        if pol in ("sac", "ppo"):
            from stable_baselines3 import PPO, SAC
            run_dir = RESULTS / (args.run_id or f"{pol}_seed42")
            model_path = run_dir / "final_model.zip"
            if not model_path.is_file():
                raise FileNotFoundError(f"model not found: {model_path}")
            model = (SAC if pol == "sac" else PPO).load(str(model_path), device="cpu")
            stats_path = run_dir / "vecnormalize_stats.json"
            if stats_path.is_file():
                norm_stats = json.loads(stats_path.read_text())
            rep["model_path"] = str(model_path.relative_to(RL05))

        def normalize(obs: np.ndarray) -> np.ndarray:
            if not norm_stats:
                return obs
            mean = np.asarray(norm_stats["obs_mean"], dtype=np.float64)
            var = np.asarray(norm_stats["obs_var"], dtype=np.float64)
            eps = float(norm_stats.get("epsilon", 1e-8))
            clip = float(norm_stats.get("clip_obs", 10.0))
            return np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)

        rows: list[dict] = []
        trace_rows: list[dict] = []
        t0 = time.perf_counter()

        for i, target in enumerate(targets):
            seed = args.seed + i
            if pol == "classical":
                # 直接 DLS：把最终目标作为笛卡尔目标
                sim.hard_reset_joints(READY)
                res = reach_rollout(sim, ctrl, target, max_steps=cfg.max_control_steps,
                                    success_tol=cfg.success_tol, hold_steps=cfg.hold_steps,
                                    settle_steps=0, trace=True)
                # 用与环境一致的奖励函数给经典基线算"可比回报"
                prev_a = np.zeros(3)
                prev_d = res.dist_trace[0] if res.dist_trace else None
                ret = 0.0
                smin = 0.0
                sat_steps = 0
                limit_steps = 0
                for k, (d, ad, sat, lim) in enumerate(zip(
                        res.dist_trace, res.applied_deltas, res.saturation_flags, res.limit_flags)):
                    a = np.clip(np.asarray(ad, dtype=np.float64) / a_scale, -1.0, 1.0)
                    term = {}
                    term["progress"] = (prev_d - d) if prev_d is not None else 0.0
                    term["action"] = -cfg.action_penalty_coef * float(np.sum(a ** 2))
                    term["smoothness"] = -cfg.smoothness_coef * float(np.sum((a - prev_a) ** 2))
                    term["joint_limit"] = -cfg.joint_limit_penalty if lim else 0.0
                    ret += sum(term.values())
                    prev_a = a
                    prev_d = d
                    smin += float(np.sum((a - prev_a) ** 2))
                    sat_steps += int(sat)
                    limit_steps += int(lim)
                if res.success:
                    ret += cfg.success_bonus
                rows.append({
                    "policy": "CLASSICAL_DLS", "episode": i, "seed": seed,
                    "success": int(res.success), "final_error": res.final_error,
                    "steps": res.steps, "time_to_success": res.time_s if res.success else "",
                    "return": ret, "action_saturation_ratio": sat_steps / max(res.steps, 1),
                    "mean_action_norm": float(np.mean(res.action_norms)) if res.action_norms else 0.0,
                    "smoothness_rms": float(np.sqrt(np.mean(np.diff([0.0] + res.action_norms) ** 2)))
                    if len(res.action_norms) > 1 else 0.0,
                    "joint_limit_violations": limit_steps,
                    "collision_steps": res.collision_steps,
                })
                for k, (d, ee) in enumerate(zip(res.dist_trace, res.ee_trace)):
                    trace_rows.append({"episode": i, "seed": seed, "step": k,
                                       "dist_m": d, "ee_x": ee[0], "ee_y": ee[1], "ee_z": ee[2],
                                       "target_x": target[0], "target_y": target[1], "target_z": target[2]})
            else:
                # 环境驱动：random / RL
                env = FR3ReachEnv(sim, cfg, target_pool=targets, seed=seed, record_trace=True)
                env.reset(seed=seed, options={"target": target})
                rng = np.random.default_rng(seed)
                a_prev = np.zeros(3)
                ret = 0.0
                smin = 0.0
                sat = 0
                limit_steps = 0
                coll = 0
                anorms = []
                while True:
                    if pol == "random":
                        a = rng.uniform(-1, 1, 3).astype(np.float32)
                    elif pol == "oracle":
                        # 上界参考策略：直接朝目标方向走满动作（不使用任何学习）
                        # 用途：判定"环境 + 奖励 + 动作接口"是否自洽 —— 若它高分，
                        # 说明任务本身可解，RL 的失败只是样本效率/优化问题。
                        err = target - sim.get_ee_pos()
                        nrm = float(np.linalg.norm(err))
                        a = (err / nrm).astype(np.float32) if nrm > 1e-9 else np.zeros(3, dtype=np.float32)
                    else:
                        obs = normalize(env._obs())
                        a, _ = model.predict(obs, deterministic=True)
                        a = np.asarray(a, dtype=np.float32).reshape(3)
                    _, r, term, trunc, info = env.step(a)
                    ret += r
                    smin += float(np.sum((np.asarray(a) - a_prev) ** 2))
                    a_prev = np.asarray(a)
                    sat += int(info["action_saturated"])
                    limit_steps += int(info["joint_limit_violation"])
                    coll += int(contact_force_norm(sim) > 1.0)
                    anorms.append(float(np.linalg.norm(a)))
                    if term or trunc:
                        break
                n = len(env.trace)
                rows.append({
                    "policy": pol.upper(), "episode": i, "seed": seed,
                    "success": int(info["success"]), "final_error": info["distance"],
                    "steps": n, "time_to_success": (n * cfg.control_dt) if info["success"] else "",
                    "return": ret,
                    "action_saturation_ratio": sat / max(n, 1),
                    "mean_action_norm": float(np.mean(anorms)) if anorms else 0.0,
                    "smoothness_rms": float(np.sqrt(smin / max(n, 1))),
                    "joint_limit_violations": limit_steps,
                    "collision_steps": coll,
                })
                for t in env.trace:
                    trace_rows.append({"episode": i, "seed": seed, "step": t["step"],
                                       "dist_m": t["dist_m"], "ee_x": t["ee_x"], "ee_y": t["ee_y"],
                                       "ee_z": t["ee_z"], "target_x": t["target_x"],
                                       "target_y": t["target_y"], "target_z": t["target_z"]})
            if (i + 1) % 10 == 0:
                print(f"[eval:{pol}] {i+1}/{len(targets)} done", flush=True)

        wall = time.perf_counter() - t0

        succ = [r for r in rows if r["success"] == 1]
        succ_times = [r["time_to_success"] for r in succ if r["time_to_success"] != ""]
        fe = np.array([r["final_error"] for r in rows], dtype=np.float64)
        metrics = {
            "policy": pol,
            "n_episodes": len(rows),
            "success_rate": len(succ) / max(len(rows), 1),
            "success_count": len(succ),
            "mean_final_position_error_m": float(fe.mean()),
            "std_final_position_error_m": float(fe.std(ddof=1)) if len(fe) > 1 else 0.0,
            "median_final_position_error_m": float(np.median(fe)),
            "mean_time_to_success_s": float(np.mean(succ_times)) if succ_times else None,
            "std_time_to_success_s": float(np.std(succ_times, ddof=1)) if len(succ_times) > 1 else None,
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "std_return": float(np.std([r["return"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "mean_episode_length": float(np.mean([r["steps"] for r in rows])),
            "action_saturation_ratio": float(np.mean([r["action_saturation_ratio"] for r in rows])),
            "mean_action_norm": float(np.mean([r["mean_action_norm"] for r in rows])),
            "action_smoothness_rms": float(np.mean([r["smoothness_rms"] for r in rows])),
            "total_joint_limit_violations": int(sum(r["joint_limit_violations"] for r in rows)),
            "total_collision_steps": int(sum(r["collision_steps"] for r in rows)),
            "eval_wall_s": round(wall, 2),
            "evaluation_tolerances": {
                "success_tol_m": cfg.success_tol, "hold_steps": cfg.hold_steps,
                "max_control_steps": cfg.max_control_steps, "control_hz": cfg.control_hz,
                "action_scale_m": cfg.action_scale,
                "upright_equivalent": "|ee - target| <= 0.02 m held for 3 control steps",
            },
        }
        if pol in ("sac", "ppo"):
            metrics["model_path"] = rep.get("model_path")

        (out_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with open(out_dir / "eval_episodes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(out_dir / "eval_trace.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            w.writerows(trace_rows)

        rep["metrics"] = metrics
        rep["success"] = True
        (out_dir / "eval_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
        print(f"EVAL_OK {pol} success_rate={metrics['success_rate']:.3f} "
              f"final_err={metrics['mean_final_position_error_m']:.4f} "
              f"return={metrics['mean_return']:.2f} wall={wall:.1f}s", flush=True)
        return 0
    except Exception as exc:
        rep["error"] = str(exc)
        rep["traceback"] = traceback.format_exc()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
        print(f"EVAL_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
