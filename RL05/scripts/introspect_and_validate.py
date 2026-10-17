#!/usr/bin/env python3
"""RL05 —— Isaac Sim FR3 接口自省与运动学验证（一次运行拿到全部信息）。

目的：不同 Isaac 版本/封装（core.prims.Articulation vs SingleArticulation）暴露的
API 不同，本脚本把候选 API 全部探测一遍并落盘，避免反复试错。

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    ./python.sh <RL05>/scripts/introspect_and_validate.py
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
OUT = RL05 / "results" / "rl05" / "introspect_report.json"
USD = RL05 / "assets" / "fr3_no_hand.usd"
EE_LINK = "fr3_link7"
DT = 1.0 / 120.0


def dump(obj) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    rep: dict = {"utc": datetime.now(timezone.utc).isoformat(), "success": False}
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})
        import omni.kit.app
        import omni.timeline
        import omni.usd

        kit = omni.kit.app.get_app()
        omni.usd.get_context().open_stage(str(USD))
        for _ in range(40):
            kit.update()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(20):
            kit.update()
        rep["stage_opened"] = True

        # ---- Articulation 候选 ----
        art = None
        for path_cls, ctor in (
            ("core.prims.Articulation", lambda: __import__("isaacsim.core.prims", fromlist=["Articulation"]).Articulation("/fr3/root_joint")),
            ("core.prims.SingleArticulation", lambda: __import__("isaacsim.core.prims", fromlist=["SingleArticulation"]).SingleArticulation("/fr3/root_joint")),
        ):
            try:
                art = ctor()
                art.initialize()
                rep["articulation_class"] = path_cls
                rep["valid"] = bool(art.is_physics_handle_valid())
                break
            except Exception as exc:
                rep.setdefault("articulation_ctor_errors", {})[path_cls] = str(exc)
        if art is None:
            raise RuntimeError("no Articulation class worked")

        rep["api_methods"] = sorted(m for m in dir(art) if not m.startswith("__"))
        rep["dof_names"] = list(art.dof_names)

        # link 名称来源
        for attr in ("link_names", "body_names"):
            if hasattr(art, attr):
                try:
                    rep[f"art.{attr}"] = list(getattr(art, attr))
                except Exception as exc:
                    rep[f"art.{attr}_error"] = str(exc)

        # ---- Jacobian ----
        jac_raw = np.asarray(art.get_jacobians(), dtype=np.float64)
        rep["jacobians_raw_shape"] = list(jac_raw.shape)
        jac = jac_raw.reshape(-1, jac_raw.shape[-2], jac_raw.shape[-1])
        rep["jacobians_links"] = int(jac.shape[0])
        rep["jacobian_nonzero_rows"] = int(np.sum(np.abs(jac).reshape(jac.shape[0], -1).max(axis=1) > 1e-9))

        # ---- link 位置来源 ----
        candidates = {}
        for meth in ("get_link_world_poses", "get_world_poses"):
            if hasattr(art, meth):
                try:
                    out = getattr(art, meth)()
                    p = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float64)
                    candidates[meth] = list(p.shape)
                except Exception as exc:
                    candidates[meth] = f"error: {exc}"
        try:
            tf = art._physics_view.get_link_transforms()
            candidates["_physics_view.get_link_transforms"] = list(np.asarray(tf).shape)
        except Exception as exc:
            candidates["_physics_view.get_link_transforms"] = f"error: {exc}"
        rep["pose_candidates"] = candidates

        def read_links() -> np.ndarray | None:
            try:
                tf = np.asarray(art._physics_view.get_link_transforms(), dtype=np.float64)
                if tf.ndim == 3:
                    tf = tf[0]
                return tf.reshape(-1, 7)[:, :3]
            except Exception:
                return None

        pos = read_links()
        rep["link_positions_shape"] = list(pos.shape) if pos is not None else None

        # ee 索引
        ln = rep.get("art.link_names") or rep.get("art.body_names") or []
        if EE_LINK in ln:
            ee_idx = ln.index(EE_LINK)
            rep["ee_index_source"] = "link_names"
        else:
            ee_idx = None
            # 用 USD 里 fr3_link7 的排序猜测不可靠，改成通过比较位置找 link7
        rep["ee_index"] = ee_idx

        if pos is not None and ee_idx is None:
            # 用 USD 路径顺序作为兜底：articulation 的 link 顺序通常与 USD 遍历一致
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            names = []
            root = stage.GetPrimAtPath("/fr3")
            for p in Usd.PrimRange(root):
                if p.GetName().startswith("fr3_link"):
                    names.append(p.GetName())
            rep["usd_link_order_sample"] = names[:25]
            if EE_LINK in names and len(names) == pos.shape[0]:
                ee_idx = names.index(EE_LINK)
                rep["ee_index_source"] = "usd_traversal"
                rep["ee_index"] = ee_idx

        if ee_idx is None:
            ee_idx = jac.shape[0] - 1
            rep["ee_index_source"] = "last_jacobian_link"
            rep["ee_index"] = ee_idx

        # ---- 找到一个"非零"的 Jacobian link 用于核验 ----
        norms = np.abs(jac[:, :3, :]).reshape(jac.shape[0], -1).max(axis=1)
        rep["jacobian_link_norms_top5"] = sorted(
            [{"link": i, "max_abs": float(norms[i])} for i in range(len(norms))],
            key=lambda d: -d["max_abs"])[:5]

        # ---- Jacobian 数值核验（用 ee_idx，同时用最大范数 link 再验一次）----
        import omni.physx
        physx = omni.physx.get_physx_interface()
        sim_t = float(timeline.get_current_time())

        def step(n: int):
            nonlocal sim_t
            for _ in range(n):
                sim_t += DT
                physx.update_simulation(DT, sim_t)
                physx.update_transformations(True, True)

        q0 = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        rep["q0"] = q0.tolist()
        step(60)

        checks = []
        for idx in {ee_idx, int(np.argmax(norms))}:
            p_before = read_links()[idx]
            Jp = jac[idx][:3, :]
            dq = np.zeros(7); dq[0] = 0.05
            art.set_joint_position_targets(q0 + dq)
            step(80)
            p_after = read_links()[idx]
            meas = p_after - p_before
            pred = Jp @ dq
            checks.append({
                "link_index": idx,
                "measured_delta": meas.tolist(),
                "predicted_delta": pred.tolist(),
                "measured_norm": float(np.linalg.norm(meas)),
                "predicted_norm": float(np.linalg.norm(pred)),
                "cos_sim": float(np.dot(meas, pred) / max(np.linalg.norm(meas) * np.linalg.norm(pred), 1e-12)),
            })
            art.set_joint_position_targets(q0)
            step(120)
        rep["jacobian_checks"] = checks

        # ---- 吞吐 ----
        n = 1500
        t0 = time.perf_counter(); step(n); t1 = time.perf_counter()
        rep["throughput"] = {"steps": n, "wall_s": round(t1 - t0, 4),
                             "steps_per_s": round(n / max(t1 - t0, 1e-9), 1)}

        rep["success"] = True
        dump(rep)
        print("INTRO_OK")
        print("INTRO_ART", rep.get("articulation_class"), "EE_IDX", rep.get("ee_index"), rep.get("ee_index_source"))
        print("INTRO_JAC", rep["jacobians_raw_shape"])
        print("INTRO_CHECKS", json.dumps(checks, default=str)[:800])
        return 0
    except Exception as exc:
        rep["error"] = str(exc)
        rep["traceback"] = traceback.format_exc()
        dump(rep)
        print(f"INTRO_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        # 强制退出，避免 Kit 关闭阶段挂起
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
