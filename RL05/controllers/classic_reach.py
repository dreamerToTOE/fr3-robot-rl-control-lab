"""RL05 —— 经典 DLS Jacobian/IK 到达控制器（Baseline）。

与 RL 使用**完全相同**的底层控制链路：
    目标误差 e → DLS 求关节速度指令 qdot → 积分成关节位置目标 → 物理步进 decimation 步

区别只在"目标"从哪来：
    Classic : x_target = 回合最终目标（一次性给出）
    RL      : x_target = 当前末端位置 + Actor 输出的单步增量

这样 RL 与经典控制的对比才是公平的（同样的 IK、同样的物理、同样的控制频率）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Q_MIN = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
Q_MAX = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])


@dataclass
class RolloutResult:
    success: bool
    final_error: float
    steps: int
    time_s: float
    joint_limit_violations: int
    collision_steps: int
    action_saturation_steps: int
    action_norms: list[float] = field(default_factory=list)
    dist_trace: list[float] = field(default_factory=list)
    ee_trace: list[list[float]] = field(default_factory=list)
    qdot_norms: list[float] = field(default_factory=list)
    applied_deltas: list[list[float]] = field(default_factory=list)
    limit_flags: list[bool] = field(default_factory=list)
    saturation_flags: list[bool] = field(default_factory=list)


_CONTACT_GETTER_CACHE: dict[int, object] = {}


def _resolve_contact_getter(sim):
    """探测可用的接触力 API 并缓存（避免每步都抛异常）。"""
    key = id(sim)
    if key in _CONTACT_GETTER_CACHE:
        return _CONTACT_GETTER_CACHE[key]
    getter = None
    for cand in (
        lambda: sim.art.get_net_contact_forces(),
        lambda: sim.art._physics_view.get_net_contact_forces(),
        lambda: sim.art._physics_view.get_net_contact_forces(clone=False),
    ):
        try:
            cand()
            getter = cand
            break
        except Exception:
            continue
    _CONTACT_GETTER_CACHE[key] = getter
    return getter


def contact_force_norm(sim) -> float:
    """所有 link 的净接触力范数最大值（用于碰撞判定）。

    Isaac Sim 4.5 的 core.prims.Articulation 未暴露 get_net_contact_forces，
    需回落到 physics view；都不可用时返回 0.0（视为无碰撞信号）。
    """
    getter = _resolve_contact_getter(sim)
    if getter is None:
        return 0.0
    try:
        f = np.asarray(getter(), dtype=np.float64)
        if f.size:
            return float(np.max(np.linalg.norm(f.reshape(-1, 3), axis=1)))
    except Exception:
        return 0.0
    return 0.0


def contact_api_available(sim) -> bool:
    return _resolve_contact_getter(sim) is not None


def reach_rollout(
    sim,
    controller,
    target: np.ndarray,
    max_steps: int = 100,
    success_tol: float = 0.02,
    hold_steps: int = 3,
    settle_steps: int | None = None,
    collision_force_threshold: float = 1.0,
    trace: bool = True,
) -> RolloutResult:
    """执行一次经典 DLS 到达 rollout。

    参数
    ----
    sim        : FR3ReachSim（已初始化）
    controller : DLSIKController
    target     : (3,) 世界系目标位置 (m)
    """
    cfg = sim.cfg
    target = np.asarray(target, dtype=np.float64).reshape(3)

    if settle_steps is not None:
        sim.step_physics(settle_steps)

    q, _ = sim.get_joint_state()
    ee = sim.get_ee_pos()
    prev_dist = float(np.linalg.norm(target - ee))

    hold = 0
    success = False
    limit_violations = 0
    collision_steps = 0
    sat_steps = 0
    action_norms: list[float] = []
    dist_trace: list[float] = []
    ee_trace: list[list[float]] = []
    qdot_norms: list[float] = []
    steps = 0

    for k in range(max_steps):
        q, _ = sim.get_joint_state()
        ee = sim.get_ee_pos()
        J = sim.get_position_jacobian()

        # 经典控制：直接把最终目标作为本步的笛卡尔目标
        res = controller.solve(q, J, ee, target, cfg.control_dt)
        sim.set_joint_targets(res.q_target)
        sim.step_physics(cfg.decimation)
        steps = k + 1

        q2, _ = sim.get_joint_state()
        ee2 = sim.get_ee_pos()
        dist = float(np.linalg.norm(target - ee2))

        if trace:
            dist_trace.append(dist)
            ee_trace.append(ee2.tolist())
            action_norms.append(float(np.linalg.norm(res.applied_delta)))
            qdot_norms.append(float(np.linalg.norm(res.qdot_cmd)))

        if res.saturated:
            sat_steps += 1
        if bool(np.any(q2 < Q_MIN - 1e-3) or np.any(q2 > Q_MAX + 1e-3)):
            limit_violations += 1
        if contact_force_norm(sim) > collision_force_threshold:
            collision_steps += 1

        if dist <= success_tol:
            hold += 1
        else:
            hold = 0
        if hold >= hold_steps:
            success = True
            break
        prev_dist = dist

    final_error = dist_trace[-1] if dist_trace else float(np.linalg.norm(target - sim.get_ee_pos()))
    return RolloutResult(
        success=success,
        final_error=final_error,
        steps=steps,
        time_s=steps * cfg.control_dt,
        joint_limit_violations=limit_violations,
        collision_steps=collision_steps,
        action_saturation_steps=sat_steps,
        action_norms=action_norms,
        dist_trace=dist_trace,
        ee_trace=ee_trace,
        qdot_norms=qdot_norms,
    )
