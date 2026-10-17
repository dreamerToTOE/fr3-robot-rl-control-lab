#!/usr/bin/env python3
"""RL05 —— 生成并验证可达目标池（用经典 DLS 控制器做可达性判定）。

原则（任务要求）："不能以不可达目标造成的失败来评价 RL 算法"。
因此所有正式使用的目标点都先用经典控制器验证过可达，再固定下来：
    - 训练目标池 (train) ：默认 600 个可达点
    - 评估目标集 (eval)  ：默认 100 个可达点，与训练池**不重叠**，且固定不变

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    ./python.sh <RL05>/scripts/generate_targets.py
"""

from __future__ import annotations

import argparse
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

READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--eval", type=int, default=100)
    ap.add_argument("--max-candidates", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--max-steps", type=int, default=140)
    args = ap.parse_args()

    out_dir = RL05 / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = RL05 / "results" / "rl05" / "target_generation.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rep: dict = {"utc": datetime.now(timezone.utc).isoformat(), "success": False}
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})

        from fr3_reach_env import FR3ReachSim, ReachConfig
        from dls_ik import default_fr3_controller
        from classic_reach import reach_rollout

        cfg = ReachConfig()
        sim = FR3ReachSim(cfg=cfg)
        ctrl = default_fr3_controller(cfg.control_dt)

        sim.hard_reset_joints(READY)
        ee_home = sim.get_ee_pos()
        rep["ee_home"] = ee_home.tolist()
        rep["control_dt"] = cfg.control_dt
        rep["control_hz"] = cfg.control_hz

        rng = np.random.default_rng(args.seed)
        reachable: list[list[float]] = []
        tried = 0
        accepted = 0
        t0 = time.perf_counter()
        need = args.train + args.eval

        while len(reachable) < need and tried < args.max_candidates:
            tried += 1
            # 在工作空间内采样：以 home 为球心，半径 0.06~0.40 m
            r = float(rng.uniform(0.06, 0.40))
            d = rng.normal(size=3)
            d /= max(np.linalg.norm(d), 1e-9)
            d[0] = abs(d[0]) * 0.9 + 0.1      # 偏向 +x（机器人前方）
            d = d / max(np.linalg.norm(d), 1e-9)
            cand = ee_home + d * r
            if cand[2] < 0.08:                 # 不要贴地
                continue
            if cand[2] > 0.95:
                continue

            sim.hard_reset_joints(READY)
            res = reach_rollout(sim, ctrl, cand, max_steps=args.max_steps,
                                success_tol=cfg.success_tol, hold_steps=cfg.hold_steps,
                                settle_steps=0, trace=False)
            if res.success:
                accepted += 1
                reachable.append([float(c) for c in cand])
                if accepted % 100 == 0:
                    print(f"[targets] accepted={accepted} tried={tried}", flush=True)

        wall = time.perf_counter() - t0
        rep["tried_candidates"] = tried
        rep["accepted"] = accepted
        rep["accept_rate"] = accepted / max(tried, 1)
        rep["generation_wall_s"] = round(wall, 2)

        if accepted < need:
            rep["error"] = f"only {accepted} reachable targets found (< {need})"
            summary_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            print(f"TARGETS_ERROR {rep['error']}", file=sys.stderr)
            return 1

        arr = np.asarray(reachable, dtype=np.float64)
        # 打乱后切分，保证 eval 与 train 不重叠
        perm = rng.permutation(len(arr))
        arr = arr[perm]
        train = arr[: args.train]
        ev = arr[args.train: args.train + args.eval]

        (out_dir / "targets_train.json").write_text(json.dumps({
            "count": int(train.shape[0]), "control_dt": cfg.control_dt,
            "success_tol": cfg.success_tol, "hold_steps": cfg.hold_steps,
            "targets": train.tolist(),
        }, indent=2), encoding="utf-8")
        (out_dir / "targets_eval.json").write_text(json.dumps({
            "count": int(ev.shape[0]), "control_dt": cfg.control_dt,
            "success_tol": cfg.success_tol, "hold_steps": cfg.hold_steps,
            "eval_seeds": [1000 + i for i in range(int(ev.shape[0]))],
            "targets": ev.tolist(),
        }, indent=2), encoding="utf-8")

        rep["train_count"] = int(train.shape[0])
        rep["eval_count"] = int(ev.shape[0])
        rep["bbox"] = {
            "x": [float(arr[:, 0].min()), float(arr[:, 0].max())],
            "y": [float(arr[:, 1].min()), float(arr[:, 1].max())],
            "z": [float(arr[:, 2].min()), float(arr[:, 2].max())],
        }
        rep["success"] = True
        summary_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print("TARGETS_OK", rep["train_count"], rep["eval_count"], "accept_rate",
              round(rep["accept_rate"], 3), "wall", rep["generation_wall_s"])
        return 0
    except Exception as exc:
        rep["error"] = str(exc)
        rep["traceback"] = traceback.format_exc()
        summary_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"TARGETS_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
