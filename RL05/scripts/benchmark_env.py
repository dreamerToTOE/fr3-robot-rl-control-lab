#!/usr/bin/env python3
"""RL05 —— 环境单步开销基准（决定 RL 训练预算是否可行）。

逐项测量每个操作的平均耗时，定位瓶颈。

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    ./python.sh <RL05>/scripts/benchmark_env.py
"""

from __future__ import annotations

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
OUT = RL05 / "results" / "rl05" / "benchmark.json"
READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


def main() -> int:
    rep: dict = {"utc": datetime.now(timezone.utc).isoformat(), "success": False}
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})
        print("BM_APP_READY", flush=True)

        from fr3_reach_env import FR3ReachSim, ReachConfig
        from dls_ik import DLSConfig, DLSIKController
        from classic_reach import contact_force_norm

        cfg = ReachConfig()
        t0 = time.perf_counter()
        sim = FR3ReachSim(cfg=cfg)
        rep["sim_init_s"] = round(time.perf_counter() - t0, 2)
        print("BM_SIM_READY", rep["sim_init_s"], flush=True)

        ctrl = DLSIKController(
            __import__("fr3_reach_env", fromlist=["Q_MIN"]).Q_MIN,
            __import__("fr3_reach_env", fromlist=["Q_MAX"]).Q_MAX,
            DLSConfig(kp=1.0, damping=0.05, qdot_max=2.62, ee_step_max=0.02, max_dq_per_step=0.05),
        )

        def timeit(fn, n=200):
            t = time.perf_counter()
            for _ in range(n):
                fn()
            return round((time.perf_counter() - t) / n * 1000.0, 4)  # ms

        sim.hard_reset_joints(READY)
        q, qd = sim.get_joint_state()

        # 单项基准
        rep["ms_get_joint_positions"] = timeit(lambda: sim.art.get_joint_positions(), 300)
        rep["ms_get_joint_velocities"] = timeit(lambda: sim.art.get_joint_velocities(), 300)
        rep["ms_get_link_transforms"] = timeit(sim.get_link_positions, 300)
        rep["ms_get_jacobians"] = timeit(sim.art.get_jacobians, 100)
        rep["ms_get_contact_forces"] = timeit(
            lambda: contact_force_norm(sim), 100)
        rep["ms_update_sim_1step"] = timeit(lambda: sim.step_physics(1), 500)
        rep["ms_update_sim_4step"] = timeit(lambda: sim.step_physics(4), 200)
        rep["ms_ik_solve"] = timeit(
            lambda: ctrl.solve(q, sim.get_position_jacobian(), sim.get_ee_pos(), sim.get_ee_pos() + np.array([0.01, 0, 0]), cfg.control_dt), 200)

        # 完整控制步（env.step 的底层）
        def one_control_step():
            ee = sim.get_ee_pos()
            J = sim.get_position_jacobian()
            qq, _ = sim.get_joint_state()
            r = ctrl.solve(qq, J, ee, ee + np.array([0.005, 0.0, 0.0]), cfg.control_dt)
            sim.set_joint_targets(r.q_target)
            sim.step_physics(cfg.decimation)

        rep["ms_full_control_step"] = timeit(one_control_step, 100)
        print("BM_PER_OPS", json.dumps({k: v for k, v in rep.items() if k.startswith("ms_")}), flush=True)

        # 端到端：一次经典 reach rollout
        from classic_reach import reach_rollout, contact_api_available
        rep["contact_api_available"] = bool(contact_api_available(sim))
        target = sim.get_ee_pos() + np.array([0.15, 0.05, -0.05])
        t0 = time.perf_counter()
        res = reach_rollout(sim, ctrl, target, max_steps=100, settle_steps=0, trace=True)
        wall = time.perf_counter() - t0
        rep["rollout"] = {"wall_s": round(wall, 2), "steps": res.steps,
                          "success": res.success, "final_error": res.final_error,
                          "ms_per_control_step": round(wall / max(res.steps, 1) * 1000, 3)}

        # 端到端：模拟目标生成的内层循环（reset + 完整 rollout）若干次
        import numpy as _np
        rng = _np.random.default_rng(0)
        ee_home = sim.get_ee_pos()
        n_cand = 20
        t0 = time.perf_counter()
        for _ in range(n_cand):
            d = rng.normal(size=3); d /= _np.linalg.norm(d)
            cand = ee_home + d * 0.25
            sim.hard_reset_joints(READY)
            reach_rollout(sim, ctrl, cand, max_steps=140, settle_steps=0, trace=False)
        per_cand = (time.perf_counter() - t0) / n_cand
        rep["candidate_loop"] = {"n": n_cand, "s_per_candidate": round(per_cand, 4)}
        rep["projected"] = {
            "control_steps_per_s": round(1.0 / max(wall / max(res.steps, 1), 1e-9), 1),
            "wall_s_for_20k_env_steps": round(20000 * (wall / max(res.steps, 1)), 1),
            "wall_s_for_50k_env_steps": round(50000 * (wall / max(res.steps, 1)), 1),
            "wall_s_for_2000_candidates": round(2000 * per_cand, 1),
        }
        rep["success"] = True
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("BM_OK", json.dumps({k: rep[k] for k in ("ms_full_control_step", "rollout", "projected")}), flush=True)
        return 0
    except Exception as exc:
        rep["error"] = str(exc)
        rep["traceback"] = traceback.format_exc()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"BM_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
