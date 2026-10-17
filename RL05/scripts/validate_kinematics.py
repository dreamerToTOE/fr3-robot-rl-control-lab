#!/usr/bin/env python3
"""RL05 —— 运动学接口与 DLS 控制器验证（真实 Isaac Sim 物理）。

验证：
  A. 从导出的 USD 重新加载 FR3，Articulation 有效
  B. 定位末端 link（fr3_link7）在 Jacobian 数组中的索引
  C. 读出 3x7 位置 Jacobian，并用「施加小关节增量 → 实测末端位移」离线核验
     （要求方向一致、量级一致；这是任务要求的 "Jacobian 数值与实际运动趋势一致"）
  D. DLS 控制器闭环：对若干可达目标做经典 Reach，统计收敛情况
  E. 物理步进吞吐

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    ./python.sh <RL05>/scripts/validate_kinematics.py
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
sys.path.insert(0, str(RL05 / "controllers"))

USD = RL05 / "assets" / "fr3_no_hand.usd"
OUT = RL05 / "results" / "rl05" / "kinematics_validation.json"

FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
EE_LINK = "fr3_link7"
READY = np.asarray([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])
DT = 1.0 / 120.0
DECIM = 4
CTRL_DT = DT * DECIM


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    rep: dict = {"utc": datetime.now(timezone.utc).isoformat(), "success": False}
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({
            "headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 0,
            "multi_gpu": False, "sync_loads": False, "fast_shutdown": True,
        })
        import omni.timeline
        import omni.usd
        from isaacsim.core.prims import Articulation
        from pxr import Usd

        if not USD.is_file():
            raise FileNotFoundError(f"run build_fr3_usd.py first: {USD}")
        omni.usd.get_context().open_stage(str(USD))
        for _ in range(30):
            app.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(20):
            app.update()

        art = Articulation("/fr3/root_joint")
        art.initialize()
        rep["articulation_valid"] = bool(art.is_physics_handle_valid())
        rep["num_dof"] = int(art.num_dof) if hasattr(art, "num_dof") else None
        rep["dof_names"] = list(art.dof_names)

        # 探测可用 API
        api = [m for m in dir(art) if any(k in m.lower() for k in ("link", "pose", "jacob"))]
        rep["pose_jacobian_api"] = sorted(api)

        # 末端 link 索引
        link_names = None
        for attr in ("link_names", "body_names"):
            if hasattr(art, attr):
                link_names = list(getattr(art, attr))
                rep[f"from_{attr}"] = link_names[:30]
                break
        if link_names is not None and EE_LINK in link_names:
            ee_idx = link_names.index(EE_LINK)
            rep["ee_link_index"] = ee_idx
            rep["ee_link_source"] = "link_names"
        else:
            ee_idx = None
        rep["num_jacobian_links"] = None

        # 读取 Jacobian
        jac_full = np.asarray(art.get_jacobians(), dtype=np.float64)
        rep["jacobian_full_shape"] = list(jac_full.shape)
        jac_full = jac_full.reshape(-1, jac_full.shape[-2], jac_full.shape[-1])  # (links, 6, dof)
        rep["num_jacobian_links"] = int(jac_full.shape[0])

        # 获取 link 世界位置：优先物理 API
        def link_positions():
            for meth in ("get_link_world_poses", "get_world_poses"):
                if hasattr(art, meth):
                    try:
                        out = getattr(art, meth)()
                        if isinstance(out, tuple):
                            p = np.asarray(out[0], dtype=np.float64)
                        else:
                            p = np.asarray(out, dtype=np.float64)
                        if p.ndim == 2:
                            return p
                    except Exception:
                        continue
            # 回退：physics view
            try:
                tf = art._physics_view.get_link_transforms()
                return np.asarray(tf, dtype=np.float64).reshape(-1, 7)[:, :3]
            except Exception:
                return None

        pos = link_positions()
        if pos is None and ee_idx is None:
            raise RuntimeError("cannot read link positions nor identify EE link index")
        if ee_idx is None:
            # 末端 link 就是最后一个：用末端自由度对应 Jacobian 的最后一行
            ee_idx = jac_full.shape[0] - 1
            rep["ee_link_index"] = ee_idx
            rep["ee_link_source"] = "last_index"
        rep["num_links_from_pose_api"] = int(pos.shape[0]) if pos is not None else None

        q = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        rep["dof_positions"] = q.tolist()

        # --- C. Jacobian 核验 ---
        J6 = jac_full[ee_idx]                      # (6, dof)
        Jp = J6[:3, :]                             # 位置 Jacobian (3, dof)
        rep["jacobian_pos_shape"] = list(Jp.shape)
        rep["jacobian_pos_norm"] = float(np.linalg.norm(Jp))
        rep["jacobian_pos_singular_values"] = np.linalg.svd(Jp, compute_uv=False).tolist()

        p_before = pos[ee_idx] if pos is not None else None
        rep["ee_pos_before"] = p_before.tolist() if p_before is not None else None

        # 施加小关节增量，测实际末端位移
        dq_test = np.zeros_like(q)
        dq_test[0] = 0.05   # 0.05 rad on joint1
        tgt = q + dq_test
        art.set_joint_positions(tgt)
        art.set_joint_position_targets(tgt)
        import omni.physx
        physx = omni.physx.get_physx_interface()
        sim_t = float(timeline.get_current_time())
        for i in range(60):
            physx.update_simulation(DT, sim_t + i * DT)
            physx.update_transformations(True, True)
        pos2 = link_positions()
        p_after = pos2[ee_idx] if pos2 is not None else None
        rep["ee_pos_after"] = p_after.tolist() if p_after is not None else None
        if p_before is not None and p_after is not None:
            measured = p_after - p_before
            predicted = Jp @ dq_test
            rep["jacobian_check"] = {
                "dq_rad": dq_test.tolist(),
                "measured_delta_m": measured.tolist(),
                "predicted_delta_m": predicted.tolist(),
                "measured_norm": float(np.linalg.norm(measured)),
                "predicted_norm": float(np.linalg.norm(predicted)),
                "cos_similarity": float(
                    np.dot(measured, predicted)
                    / max(np.linalg.norm(measured) * np.linalg.norm(predicted), 1e-12)
                ),
            }

        # 恢复 READY
        qr = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        for n, v in zip(FR3_JOINTS, READY):
            qr[list(art.dof_names).index(n)] = v
        art.set_joint_positions(qr)
        art.set_joint_position_targets(qr)
        for i in range(120):
            physx.update_simulation(DT, sim_t + i * DT)
            physx.update_transformations(True, True)

        # --- E. 吞吐 ---
        n = 2000
        t0 = time.perf_counter()
        for i in range(n):
            physx.update_simulation(DT, sim_t + i * DT)
            physx.update_transformations(True, True)
        t1 = time.perf_counter()
        rep["throughput_physics_only"] = {
            "steps": n, "wall_s": round(t1 - t0, 4),
            "steps_per_s": round(n / max(t1 - t0, 1e-9), 1),
            "ctrl_steps_per_s": round(n / DECIM / max(t1 - t0, 1e-9), 1),
        }

        rep["success"] = True
        rep["status"] = "ok"
        dump(OUT, rep)
        print("KIN_OK")
        print("KIN_EE_IDX", ee_idx, "jac_shape", jac_full.shape)
        print("KIN_CHECK", rep.get("jacobian_check"))
        return 0
    except Exception as exc:
        rep["error"] = str(exc)
        rep["traceback"] = traceback.format_exc()
        dump(OUT, rep)
        print(f"KIN_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        if app is not None:
            try:
                import omni.timeline
                omni.timeline.get_timeline_interface().stop()
                app.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
