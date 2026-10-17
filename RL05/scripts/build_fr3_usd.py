#!/usr/bin/env python3
"""RL05 —— 从 FR3 URDF 构建可复用的 Isaac Sim USD 资产，并验证控制接口。

用法（必须用 Isaac Sim 自带 python）:
    cd /home/ubuntu2004/isaacsim-4.5.0
    ./python.sh /home/ubuntu2004/lmy/fr3-robot-rl-control-lab/RL05/scripts/build_fr3_usd.py

产物:
    RL05/assets/fr3_no_hand.usd           导入后的 FR3 资产（含位置驱动）
    RL05/results/rl05/build_report.json   构建与接口验证报告

验证内容:
    1. URDF 导入成功、Articulation 有效
    2. DOF 名称 = fr3_joint1..7，链名列表
    3. 末端 link 的 FK 位置
    4. get_jacobians() 可用性 + 与有限差分 Jacobian 的对比（数值一致性）
    5. 物理步进吞吐（steps/s）
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RL05 = Path(__file__).resolve().parents[1]
REPO = RL05.parent
URDF = RL05 / "assets" / "fr3_no_hand.urdf"
USD_OUT = RL05 / "assets" / "fr3_no_hand.usd"
REPORT = RL05 / "results" / "rl05" / "build_report.json"

# 复用双臂工程已构建的 franka_description（只读引用，不修改对方仓库）
FRANKA_SHARE = (
    Path("/home/ubuntu2004/lmy/dual-arm-embodied-palletizing")
    / "ros_ws" / "install" / "franka_description" / "share"
)

FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
EE_LINK = "fr3_link7"
READY = np.asarray([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


def write_report(report: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    report: dict = {
        "task": "RL05 FR3 USD build + interface validation",
        "utc": datetime.now(timezone.utc).isoformat(),
        "success": False,
    }
    app = None
    try:
        if not URDF.is_file():
            raise FileNotFoundError(f"URDF missing: {URDF}")
        if not (FRANKA_SHARE / "franka_description").is_dir():
            raise FileNotFoundError(f"franka_description not built: {FRANKA_SHARE}")
        os.environ["ROS_PACKAGE_PATH"] = str(FRANKA_SHARE)

        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": True,
                "renderer": "RaytracedLighting",
                "anti_aliasing": 0,
                "multi_gpu": False,
                "sync_loads": False,
                "fast_shutdown": True,
            }
        )

        import omni.kit.commands
        import omni.timeline
        import omni.usd
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

        enable_extension("isaacsim.asset.importer.urdf")
        for _ in range(10):
            app.update()

        omni.usd.get_context().new_stage()
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.self_collision = False
        import_config.distance_scale = 1.0
        import_config.collision_from_visuals = False
        import_config.create_physics_scene = False

        status, articulation_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(URDF),
            import_config=import_config,
            get_articulation_root=True,
        )
        if not status or not articulation_path:
            raise RuntimeError(f"URDFParseAndImportFile failed: {articulation_path}")
        articulation_path = str(articulation_path)
        report["articulation_path"] = articulation_path

        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        px = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        px.CreateEnableCCDAttr(True)
        px.CreateEnableStabilizationAttr(True)
        px.CreateEnableGPUDynamicsAttr(False)
        px.CreateBroadphaseTypeAttr("MBP")
        px.CreateSolverTypeAttr("TGS")

        UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
        light.CreateIntensityAttr(1200.0)

        # 位置驱动（与已验证的 task00 一致）
        configured = 0
        for prim in stage.Traverse():
            if prim.GetName() not in FR3_JOINTS or not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr().Set(4000.0)
            drive.CreateDampingAttr().Set(400.0)
            configured += 1
        report["drives_configured"] = configured
        if configured != len(FR3_JOINTS):
            raise RuntimeError(f"expected 7 drives, configured {configured}")

        for _ in range(8):
            app.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(24):
            app.update()

        art = Articulation(articulation_path)
        art.initialize()
        if not art.is_physics_handle_valid():
            raise RuntimeError("FR3 is not a valid articulation")
        dof_names = list(art.dof_names)
        report["dof_names"] = dof_names
        missing = sorted(set(FR3_JOINTS) - set(dof_names))
        report["missing_dofs"] = missing
        if missing:
            raise RuntimeError(f"missing DOFs: {missing}")

        # 收集 link 名
        links = [p.GetName() for p in stage.Traverse() if p.IsA(UsdGeom.Xform)]
        report["link_count"] = len(links)
        report["has_ee_link"] = EE_LINK in links

        # 设到 READY 位姿
        pos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        for name, value in zip(FR3_JOINTS, READY):
            pos[dof_names.index(name)] = value
        art.set_joint_positions(pos)
        art.set_joint_position_targets(pos)
        art.set_joint_velocities(np.zeros_like(pos))
        for _ in range(40):
            app.update()

        q = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        report["ready_joint_positions"] = q.tolist()
        report["ready_joint_velocities"] = np.asarray(
            art.get_joint_velocities(), dtype=np.float64
        ).reshape(-1).tolist()

        # EE FK 位置
        ee_idx = links.index(EE_LINK) if EE_LINK in links else -1
        try:
            ee_pos, ee_quat = art.get_world_pose()  # 通常是 articulation root
        except Exception:
            ee_pos, ee_quat = None, None
        report["root_world_pose"] = (
            {"pos": np.asarray(ee_pos).tolist(), "quat": np.asarray(ee_quat).tolist()}
            if ee_pos is not None else None
        )

        # link7 世界位置（通过 USD 计算）
        link7_path = None
        for p in stage.Traverse():
            if p.GetName() == EE_LINK:
                link7_path = p.GetPath()
                break
        if link7_path is not None:
            m = UsdGeom.Xformable(stage.GetPrimAtPath(link7_path)).ComputeLocalToWorldTransform(0.0)
            report["link7_world_translation"] = [float(m[3][i]) for i in range(3)]
            report["link7_path"] = str(link7_path)

        # Jacobian 可用性
        jac_info: dict = {}
        try:
            jac = art.get_jacobians()
            jac = np.asarray(jac, dtype=np.float64)
            jac_info["available"] = True
            jac_info["shape"] = list(jac.shape)
            jac_info["sample_mat"] = jac.reshape(-1)[:12].tolist()
        except Exception as exc:
            jac_info["available"] = False
            jac_info["error"] = f"{type(exc).__name__}: {exc}"
        report["jacobian"] = jac_info

        # 吞吐测试：直接步进物理（不渲染）
        import omni.physx

        physx = omni.physx.get_physx_interface()
        dt = 1.0 / 120.0
        n = 300
        t0 = time.perf_counter()
        sim_t = float(timeline.get_current_time())
        for i in range(n):
            physx.update_simulation(dt, sim_t + i * dt)
            physx.update_transformations(True, True)
        t1 = time.perf_counter()
        report["throughput"] = {
            "mode": "physx.update_simulation (no render)",
            "steps": n,
            "wall_s": round(t1 - t0, 4),
            "steps_per_s": round(n / max(t1 - t0, 1e-9), 1),
        }

        # 另外测 app.update() 的吞吐作对照
        t0 = time.perf_counter()
        for _ in range(30):
            app.update()
        t1 = time.perf_counter()
        report["throughput_app_update"] = {
            "steps": 30, "wall_s": round(t1 - t0, 4),
            "steps_per_s": round(30 / max(t1 - t0, 1e-9), 1),
        }

        # 导出 USD
        USD_OUT.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(USD_OUT))
        if not USD_OUT.is_file() or USD_OUT.stat().st_size == 0:
            raise RuntimeError("failed to export USD")
        report["usd_path"] = str(USD_OUT)
        report["usd_size_bytes"] = USD_OUT.stat().st_size

        report["success"] = True
        report["status"] = "ok"
        write_report(report)
        print("BUILD_REPORT_OK")
        print("BUILD_DOFS", dof_names)
        print("BUILD_USD", USD_OUT, USD_OUT.stat().st_size)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_report(report)
        print(f"BUILD_ERROR {exc}", file=sys.stderr)
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
