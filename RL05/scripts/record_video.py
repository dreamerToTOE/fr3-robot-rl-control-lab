#!/usr/bin/env python3
"""RL05 —— 录制控制演示视频（best-effort，渲染不可用则优雅跳过）。

用 omni.replicator.core 的 rgb annotator 在无界面环境下离屏渲染，写入 mp4。
录制对象：classical / sac / ppo 的典型回合；可选失败案例。

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    PYTHONPATH=<RL05>/.pydeps ./python.sh <RL05>/scripts/record_video.py --policy sac --run-id sac_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

RL05 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL05 / "envs"))
sys.path.insert(0, str(RL05 / "controllers"))
RESULTS = RL05 / "results" / "rl05"
VIDEOS = RESULTS / "videos"
READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["classical", "random", "sac", "ppo"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--target-index", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()

    VIDEOS.mkdir(parents=True, exist_ok=True)
    out_mp4 = VIDEOS / f"{args.policy}_reach_target{args.target_index}.mp4"
    report = {"policy": args.policy, "out": str(out_mp4), "success": False}

    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})

        import omni.kit.app
        import omni.replicator.core as rep  # noqa: F401  (触发扩展加载)

        from fr3_reach_env import FR3ReachEnv, FR3ReachSim, ReachConfig
        from dls_ik import default_fr3_controller
        from classic_reach import reach_rollout

        cfg = ReachConfig(reset_joint_noise=0.0)
        sim = FR3ReachSim(cfg=cfg)
        targets = np.asarray(json.loads((RL05 / "assets" / "targets_eval.json").read_text())["targets"])
        target = targets[args.target_index]

        # 相机（俯视偏前的第三人称视角）
        cam = rep.create.camera(position=(1.35, -0.95, 0.95), look_at=(0.40, 0.0, 0.55), focal_length=24.0)
        rp = rep.create.render_product(cam, (args.width, args.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(rp)

        kit = omni.kit.app.get_app()
        for _ in range(20):
            kit.update()

        frames: list[np.ndarray] = []

        def capture(n: int = 4):
            for _ in range(n):
                kit.update()
            try:
                data = rgb.get_data()
                if data is not None and getattr(data, "size", 0):
                    frames.append(np.asarray(data)[:, :, :3].copy())
            except Exception:
                pass

        # 模型 / 策略
        model = None
        norm = None
        if args.policy in ("sac", "ppo"):
            from stable_baselines3 import PPO, SAC
            run_dir = RESULTS / (args.run_id or f"{args.policy}_seed42")
            model = (SAC if args.policy == "sac" else PPO).load(str(run_dir / "final_model.zip"), device="cpu")
            sp = run_dir / "vecnormalize_stats.json"
            if sp.exists():
                norm = json.loads(sp.read_text())

        def normalize(obs):
            if not norm:
                return obs
            m = np.asarray(norm["obs_mean"]); v = np.asarray(norm["obs_var"])
            return np.clip((obs - m) / np.sqrt(v + float(norm.get("epsilon", 1e-8))),
                           -float(norm.get("clip_obs", 10.0)), float(norm.get("clip_obs", 10.0))).astype(np.float32)

        if args.policy == "classical":
            sim.hard_reset_joints(READY)
            capture(8)
            ctrl = default_fr3_controller(cfg.control_dt)
            max_steps = cfg.max_control_steps
            hold = 0
            for k in range(max_steps):
                q, _ = sim.get_joint_state()
                ee = sim.get_ee_pos()
                J = sim.get_position_jacobian()
                res = ctrl.solve(q, J, ee, target, cfg.control_dt)
                sim.set_joint_targets(res.q_target)
                sim.step_physics(cfg.decimation)
                capture(2)
                d = float(np.linalg.norm(target - sim.get_ee_pos()))
                hold = hold + 1 if d <= cfg.success_tol else 0
                if hold >= cfg.hold_steps:
                    break
            success = hold >= cfg.hold_steps
        else:
            env = FR3ReachEnv(sim, cfg, target_pool=targets, seed=1000, record_trace=True)
            env.reset(seed=1000, options={"target": target})
            capture(8)
            rng = np.random.default_rng(1000)
            while True:
                if args.policy == "random":
                    a = rng.uniform(-1, 1, 3).astype(np.float32)
                else:
                    a, _ = model.predict(normalize(env._obs()), deterministic=True)
                    a = np.asarray(a, dtype=np.float32).reshape(3)
                _, _, term, trunc, info = env.step(a)
                capture(2)
                if term or trunc:
                    break
            success = bool(info["success"])

        if not frames:
            print("VIDEO_SKIP no frames captured (rendering unavailable)", flush=True)
            report["skipped"] = "no frames"
            (VIDEOS / f"{args.policy}_report.json").write_text(json.dumps(report, indent=2))
            return 0

        if args.max_frames:
            frames = frames[: args.max_frames]
        try:
            import imageio.v2 as imageio
            imageio.mimsave(str(out_mp4), frames, fps=30)
        except Exception as exc:
            print(f"VIDEO_SKIP encode failed: {exc}", flush=True)
            report["skipped"] = f"encode: {exc}"
            (VIDEOS / f"{args.policy}_report.json").write_text(json.dumps(report, indent=2))
            return 0

        report.update({"success": True, "frames": len(frames),
                       "reach_success": bool(success),
                       "target": target.tolist()})
        (VIDEOS / f"{args.policy}_report.json").write_text(json.dumps(report, indent=2))
        print(f"VIDEO_OK {out_mp4} frames={len(frames)} reach_success={success}", flush=True)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        VIDEOS.mkdir(parents=True, exist_ok=True)
        (VIDEOS / f"{args.policy}_report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"VIDEO_SKIP {exc}", flush=True)
        return 0
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
