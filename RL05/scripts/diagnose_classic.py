#!/usr/bin/env python3
"""RL05 —— 经典 DLS 控制器诊断（进入 RL 训练前的门禁验证）。

诊断内容：
  1. 固定 set 的目标点（不同距离/方向）下，经典控制器的成功率与最终误差
  2. 误差是否"卡住"（平台化）→ 判断是增益不足、限幅过紧还是关节驱动跟不上
  3. 关节目标跟踪误差（指令 q_target vs 实际 q）→ 判断位置驱动是否够力
  4. 关节限位/饱和统计

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    ./python.sh <RL05>/scripts/diagnose_classic.py [--steps 200]
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
OUT = RL05 / "results" / "rl05" / "classic_diagnosis.json"
READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


def build_targets(home: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """构造一组覆盖不同方向与距离的目标点（用于诊断，不做可达性预筛）。"""
    dirs = {
        "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
        "-z": np.array([0.0, 0.0, -1.0]), "+z": np.array([0.0, 0.0, 1.0]),
        "xz": np.array([1.0, 0.0, -1.0]) / np.sqrt(2),
        "xy": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    }
    out = []
    for name, d in dirs.items():
        for r in (0.10, 0.20, 0.30):
            out.append((f"{name}_{int(r*100)}cm", home + d * r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    rep: dict = {"utc": datetime.now(timezone.utc).isoformat(), "success": False}
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})
        from fr3_reach_env import FR3ReachSim, ReachConfig, Q_MIN, Q_MAX
        from dls_ik import default_fr3_controller
        from classic_reach import contact_api_available, contact_force_norm

        cfg = ReachConfig()
        sim = FR3ReachSim(cfg=cfg)
        ctrl = default_fr3_controller(cfg.control_dt)
        rep["contact_api_available"] = bool(contact_api_available(sim))

        sim.hard_reset_joints(READY)
        home = sim.get_ee_pos()
        rep["home"] = home.tolist()
        rep["control_dt"] = cfg.control_dt

        rows = []
        t0 = time.perf_counter()
        for name, target in build_targets(home):
            sim.hard_reset_joints(READY)
            q0, _ = sim.get_joint_state()
            dist_trace = []
            track_err = []
            hold = 0
            success = False
            sat = 0
            limit = 0
            for k in range(args.steps):
                q, _ = sim.get_joint_state()
                ee = sim.get_ee_pos()
                J = sim.get_position_jacobian()
                res = ctrl.solve(q, J, ee, target, cfg.control_dt)
                sim.set_joint_targets(res.q_target)
                sim.step_physics(cfg.decimation)
                q2, _ = sim.get_joint_state()
                ee2 = sim.get_ee_pos()
                d = float(np.linalg.norm(target - ee2))
                dist_trace.append(d)
                track_err.append(float(np.max(np.abs(q2 - res.q_target))))
                sat += int(res.saturated)
                if bool(np.any(q2 < Q_MIN - 1e-3) or np.any(q2 > Q_MAX + 1e-3)):
                    limit += 1
                if d <= cfg.success_tol:
                    hold += 1
                else:
                    hold = 0
                if hold >= cfg.hold_steps:
                    success = True
                    break
            rows.append({
                "name": name,
                "target": target.tolist(),
                "target_dist_from_home": float(np.linalg.norm(target - home)),
                "success": bool(success),
                "steps": len(dist_trace),
                "final_error": dist_trace[-1] if dist_trace else None,
                "min_error": float(np.min(dist_trace)) if dist_trace else None,
                "err_at_25pct": dist_trace[len(dist_trace) // 4] if dist_trace else None,
                "err_at_50pct": dist_trace[len(dist_trace) // 2] if dist_trace else None,
                "err_at_75pct": dist_trace[3 * len(dist_trace) // 4] if dist_trace else None,
                "mean_joint_tracking_error_rad": float(np.mean(track_err)),
                "max_joint_tracking_error_rad": float(np.max(track_err)),
                "saturation_steps": sat,
                "limit_violation_steps": limit,
            })
        wall = time.perf_counter() - t0

        ok = [r for r in rows if r["success"]]
        rep["rows"] = rows
        rep["summary"] = {
            "n_targets": len(rows),
            "success_count": len(ok),
            "success_rate": len(ok) / max(len(rows), 1),
            "mean_final_error": float(np.mean([r["final_error"] for r in rows])),
            "median_final_error": float(np.median([r["final_error"] for r in rows])),
            "mean_tracking_error_rad": float(np.mean([r["mean_joint_tracking_error_rad"] for r in rows])),
            "max_tracking_error_rad": float(np.max([r["max_joint_tracking_error_rad"] for r in rows])),
            "total_limit_violation_steps": int(sum(r["limit_violation_steps"] for r in rows)),
            "wall_s": round(wall, 2),
        }
        rep["success"] = True
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("DIAG_OK", json.dumps(rep["summary"]), flush=True)
        for r in rows:
            print(f"  {r['name']:10s} succ={int(r['success'])} final={r['final_error']:.4f} "
                  f"min={r['min_error']:.4f} track={r['mean_joint_tracking_error_rad']:.4f} "
                  f"sat={r['saturation_steps']} steps={r['steps']}", flush=True)
        return 0
    except Exception as exc:
        rep["error"] = str(exc)
        rep["traceback"] = traceback.format_exc()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"DIAG_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
