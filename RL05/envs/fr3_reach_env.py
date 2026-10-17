"""RL05 —— FR3 Cartesian Reach 强化学习环境（Isaac Sim 4.5 真实物理）。

设计（对应 RL05 任务书的 MDP 定义）
------------------------------------------------------------------
Observation（20 维, float32）
    [ q (7)            关节位置, rad
      qdot (7)         关节速度, rad/s
      ee_error (3)     目标 - 当前末端位置, 世界系, m
      prev_action (3)  上一步动作（归一化, [-1,1]） ]
    注意：本环境返回的是**未归一化**的原始观测；训练时由 SB3 的 VecNormalize
    统一归一化，并把统计量随模型一起保存，保证训练/评估一致。

Action（3 维, float32, ∈[-1,1]）
    末端笛卡尔位置增量方向。实际位移 = action * action_scale（默认 0.01 m/控制步）。
    随后交给 DLS IK 控制器转成关节位置指令（不是直接输出 7 关节力矩）。

Reward（分项，全部单独记录）
    progress   = |prev_err| - |cur_err|
    success    = +success_bonus（达成并保持时，仅触发一次）
    action     = -action_penalty_coef * ||a||^2
    smoothness = -smoothness_coef * ||a - a_prev||^2
    safety     = 关节越限 / 碰撞时的惩罚（安全约束由底层限幅保证，奖励只是信号）

Termination
    success    : |err| <= success_tol 连续保持 hold_steps 个控制步
    failure    : 关节位置越过限位软阈值（不可恢复的非法状态）
    truncation : 达到 max_control_steps（默认 100，即 100/30 ≈ 3.33 s 仿真时间）

物理时间
    physics_dt = 1/120 s；decimation = 4 → 控制频率 30 Hz，每个 RL 动作 = 4 个物理步。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # 允许在没有 gymnasium 的环境里 import 本模块（仅做静态检查用）
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

RL05 = Path(__file__).resolve().parents[1]

FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
EE_LINK = "fr3_link7"

Q_MIN = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
Q_MAX = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
QDOT_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


@dataclass
class ReachConfig:
    physics_dt: float = 1.0 / 120.0
    decimation: int = 4
    max_control_steps: int = 100
    action_scale: float = 0.01          # m / 控制步（末端位移上限）
    success_tol: float = 0.02           # m
    hold_steps: int = 3                 # 连续保持的控制步数
    reset_joint_noise: float = 0.05     # rad
    settle_physics_steps: int = 60
    progress_coef: float = 1.0          # progress 项权重
    success_bonus: float = 10.0
    action_penalty_coef: float = 0.01
    smoothness_coef: float = 0.02
    joint_limit_penalty: float = 1.0
    joint_limit_margin: float = 0.0     # 越限判定余量（rad）
    ik: dict = field(default_factory=dict)

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def control_hz(self) -> float:
        return 1.0 / self.control_dt


class FR3ReachSim:
    """薄封装：Stage/Articulation/物理步进/Jacobian/末端位姿。

    必须在 SimulationApp 之后构造（本模块内部延迟 import Isaac 模块）。
    """

    def __init__(self, usd_path: Path | None = None, cfg: ReachConfig | None = None,
                 articulation_path: str = "/fr3/root_joint"):
        self.cfg = cfg or ReachConfig()
        self.usd_path = Path(usd_path or (RL05 / "assets" / "fr3_no_hand.usd"))
        self.articulation_path = articulation_path
        self._sim_time = 0.0

        import omni.timeline
        import omni.usd
        from isaacsim.core.prims import Articulation

        if not self.usd_path.is_file():
            raise FileNotFoundError(
                f"FR3 USD missing: {self.usd_path}. Run scripts/build_fr3_usd.py first."
            )
        omni.usd.get_context().open_stage(str(self.usd_path))
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        self._pump(20)

        self.art = Articulation(self.articulation_path)
        self.art.initialize()
        if not self.art.is_physics_handle_valid():
            raise RuntimeError("FR3 articulation handle invalid")
        self.dof_names = list(self.art.dof_names)

        # link 索引（Jacobian 的第 2 维按 link 名称列表排列）
        # 注意：core.prims.Articulation 没有 link_names，实际提供的是 body_names；
        # 而 get_world_poses() 只返回根位姿，逐 link 位姿必须走 physics view。
        self.link_names: list[str] = []
        for attr in ("body_names", "link_names"):
            if hasattr(self.art, attr):
                try:
                    names = list(getattr(self.art, attr))
                    if names:
                        self.link_names = names
                        break
                except Exception:
                    continue
        self.ee_idx = self.link_names.index(EE_LINK) if EE_LINK in self.link_names else -1

        jac = np.asarray(self.art.get_jacobians(), dtype=np.float64)
        self._jac_num_links = int(jac.reshape(-1, jac.shape[-2], jac.shape[-1]).shape[0]) \
            if jac.ndim >= 3 else 1
        if self.ee_idx < 0:
            self.ee_idx = self._jac_num_links - 1

        import omni.physx
        self.physx = omni.physx.get_physx_interface()

    # ------------------------------------------------------------------
    def _pump(self, n: int) -> None:
        """推进 Kit（用于加载/初始化阶段）。"""
        from isaacsim import SimulationApp

        app = SimulationApp._instance if hasattr(SimulationApp, "_instance") else None
        import omni.kit.app

        app_iface = omni.kit.app.get_app()
        for _ in range(n):
            app_iface.update()
        _ = app

    # ------------------------------------------------------------------
    def step_physics(self, n: int) -> None:
        """只推进物理（不渲染），用于 RL 高速循环。"""
        dt = self.cfg.physics_dt
        for _ in range(n):
            self._sim_time += dt
            self.physx.update_simulation(dt, self._sim_time)
            self.physx.update_transformations(True, True)

    # ------------------------------------------------------------------
    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(self.art.get_joint_positions(), dtype=np.float64).reshape(-1)
        qd = np.asarray(self.art.get_joint_velocities(), dtype=np.float64).reshape(-1)
        return q, qd

    def get_link_positions(self) -> np.ndarray | None:
        """所有 link 的世界位置 (L,3)。

        只用 physics view：Isaac Sim 4.5 中 USD 与 Fabric 是两套表示，
        USD 的 xformOp 不会随物理步进更新；get_world_poses() 又只给根位姿。
        """
        try:
            tf = np.asarray(self.art._physics_view.get_link_transforms(), dtype=np.float64)
            if tf.ndim == 3:
                tf = tf[0]
            return tf.reshape(-1, 7)[:, :3]
        except Exception:
            pass
        for meth in ("get_link_world_poses",):
            if hasattr(self.art, meth):
                try:
                    out = getattr(self.art, meth)()
                    p = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float64)
                    if p.ndim == 2 and p.shape[1] >= 3:
                        return p[:, :3]
                except Exception:
                    continue
        return None

    def get_ee_pos(self) -> np.ndarray:
        pos = self.get_link_positions()
        if pos is not None and 0 <= self.ee_idx < pos.shape[0]:
            return pos[self.ee_idx].copy()
        raise RuntimeError("cannot read end-effector position")

    def get_position_jacobian(self) -> np.ndarray:
        jac = np.asarray(self.art.get_jacobians(), dtype=np.float64)
        jac = jac.reshape(-1, jac.shape[-2], jac.shape[-1])   # (links, 6, dof)
        return jac[self.ee_idx][:3, :].copy()

    # ------------------------------------------------------------------
    def set_joint_targets(self, q_target: np.ndarray) -> None:
        t = np.asarray(q_target, dtype=np.float64).reshape(-1)
        self.art.set_joint_position_targets(t)

    def hard_reset_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.art.set_joint_positions(q)
        self.art.set_joint_position_targets(q)
        self.art.set_joint_velocities(np.zeros_like(q))
        self.step_physics(self.cfg.settle_physics_steps)


class FR3ReachEnv(gym.Env if gym is not None else object):
    """Gymnasium 风格 FR3 Reach 环境。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        sim: FR3ReachSim,
        cfg: ReachConfig | None = None,
        target_pool: np.ndarray | None = None,
        seed: int = 0,
        record_trace: bool = True,
    ):
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required for FR3ReachEnv")
        super().__init__()
        self.sim = sim
        self.cfg = cfg or sim.cfg
        self.rng = np.random.default_rng(seed)
        self.target_pool = None if target_pool is None else np.asarray(target_pool, dtype=np.float64)
        self.record_trace = record_trace

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        from dls_ik import default_fr3_config, DLSIKController

        # 注意：Kp 必须是 1/s 量纲的速度增益，见 dls_ik.default_fr3_config 的说明。
        # 这里把 ee_step_max 设为动作幅度，使「请求位移 = 实际位移」。
        _dls = default_fr3_config(self.cfg.control_dt)
        _dls.ee_step_max = self.cfg.action_scale
        self.controller = DLSIKController(Q_MIN, Q_MAX, _dls)

        # 内部状态
        self._q = READY.copy()
        self._prev_action = np.zeros(3)
        self._prev_dist = 0.0
        self._target = READY.copy()
        self._step_count = 0
        self._hold = 0
        self._reward_terms: dict[str, float] = {}
        self.trace: list[dict] = []
        self.current_target: np.ndarray | None = None

    # ------------------------------------------------------------------
    def set_target(self, target: np.ndarray, reset_joints: bool = True) -> None:
        """用于确定性评估：显式指定目标。"""
        self.current_target = np.asarray(target, dtype=np.float64).reshape(3)
        self._target = self.current_target.copy()
        if reset_joints:
            self._reset_joints_and_settle(self.current_target)

    # ------------------------------------------------------------------
    def _sample_q0(self) -> np.ndarray:
        return np.clip(
            READY + self.rng.uniform(-self.cfg.reset_joint_noise, self.cfg.reset_joint_noise, 7),
            Q_MIN, Q_MAX,
        )

    def _reset_joints_and_settle(self, target: np.ndarray) -> None:
        q0 = self._sample_q0()
        self.sim.hard_reset_joints(q0)
        self._q, _ = self.sim.get_joint_state()
        self._prev_action = np.zeros(3)
        self._prev_dist = float(np.linalg.norm(target - self.sim.get_ee_pos()))

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        if "target" in opts:
            target = np.asarray(opts["target"], dtype=np.float64).reshape(3)
        elif self.target_pool is not None:
            target = self.target_pool[self.rng.integers(len(self.target_pool))].copy()
        else:
            raise RuntimeError("no target source: provide target_pool or options['target']")

        self.current_target = target
        self._target = target.copy()      # 注意：必须在此处赋值，(3,) 目标向量
        self._step_count = 0
        self._hold = 0
        self._success_paid = False
        self._reward_terms = {}
        self.trace = []
        self._reset_joints_and_settle(target)
        return self._obs(), {"target": target.copy(), "reset_joint_noise": self.cfg.reset_joint_noise}

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        q, qd = self.sim.get_joint_state()
        err = self._target - self.sim.get_ee_pos()
        return np.concatenate([q, qd, err, self._prev_action]).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        terms: dict[str, float] = {}

        ee = self.sim.get_ee_pos()
        J = self.sim.get_position_jacobian()
        q, _ = self.sim.get_joint_state()

        # 动作 → 末端子目标（单步位移 = a * action_scale）
        x_sub = ee + a * self.cfg.action_scale
        res = self.controller.solve(q, J, ee, x_sub, self.cfg.physics_dt * self.cfg.decimation)
        self.sim.set_joint_targets(res.q_target)
        self.sim.step_physics(self.cfg.decimation)

        q2, qd2 = self.sim.get_joint_state()
        ee2 = self.sim.get_ee_pos()
        dist = float(np.linalg.norm(self._target - ee2))
        prev_dist = self._prev_dist

        # --- reward terms ---
        terms["progress"] = self.cfg.progress_coef * (prev_dist - dist)
        terms["action"] = -self.cfg.action_penalty_coef * float(np.sum(a ** 2))
        terms["smoothness"] = -self.cfg.smoothness_coef * float(np.sum((a - self._prev_action) ** 2))

        limit_viol = bool(np.any(q2 < Q_MIN - 1e-3) or np.any(q2 > Q_MAX + 1e-3))
        terms["joint_limit"] = -self.cfg.joint_limit_penalty if limit_viol else 0.0

        if dist <= self.cfg.success_tol:
            self._hold += 1
        else:
            self._hold = 0

        success = self._hold >= self.cfg.hold_steps
        terms["success"] = self.cfg.success_bonus if (success and not self._success_paid) else 0.0
        if success:
            self._success_paid = True

        reward = float(sum(terms.values()))
        self._reward_terms = terms

        # --- termination ---
        terminated = False
        truncated = False
        fail_reason = ""
        if success:
            terminated = True
        elif limit_viol:
            terminated = True
            fail_reason = "joint_limit_violation"
        elif self._step_count + 1 >= self.cfg.max_control_steps:
            truncated = True

        if self.record_trace:
            self.trace.append({
                "step": self._step_count,
                "time_s": (self._step_count + 1) * self.cfg.control_dt,
                "ee_x": float(ee2[0]), "ee_y": float(ee2[1]), "ee_z": float(ee2[2]),
                "target_x": float(self._target[0]), "target_y": float(self._target[1]),
                "target_z": float(self._target[2]),
                "dist_m": dist,
                "action_norm": float(np.linalg.norm(a)),
                "saturated": bool(res.saturated),
                "torque_proxy": float(np.linalg.norm(res.qdot_cmd)),
            })

        info = {
            "distance": dist,
            "success": bool(success),
            "is_success": bool(success),
            "is_truncated": bool(truncated),
            "failure": fail_reason,
            "reward_terms": dict(terms),
            "ee_pos": ee2.copy(),
            "target": self._target.copy(),
            "joint_limit_violation": limit_viol,
            "action_norm": float(np.linalg.norm(a)),
            "action_saturated": bool(res.saturated),
        }

        self._q = q2
        self._prev_action = a
        self._prev_dist = dist
        self._step_count += 1
        self._success_paid = self._success_paid if success else False
        if terminated or truncated:
            info["episode"] = {
                "success": bool(success),
                "final_distance": dist,
                "steps": self._step_count,
                "time_s": self._step_count * self.cfg.control_dt,
            }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self):
        pass
