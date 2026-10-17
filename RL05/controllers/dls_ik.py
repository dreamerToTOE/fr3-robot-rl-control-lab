"""RL05 —— 阻尼最小二乘（Damped Least Squares, DLS）Jacobian/IK 控制器。

这是任务要求的「经典控制 Baseline」，同时也是 RL 策略的底层执行器：
Actor 只输出末端笛卡尔位置增量（Δx, Δy, Δz），本模块负责把它变成合法的关节指令。

核心公式（任务给定）：

    e        = x_target - x_current                       # 末端位置误差 (m)
    qdot_cmd = J^T · (J·J^T + λ²·I)^(-1) · Kp · e          # 关节速度指令 (rad/s)

其中
    J    ：末端「位置」Jacobian，3×n（只取线速度行）
    λ    ：阻尼系数（奇异位形附近抑制数值爆炸）
    Kp   ：位置控制增益
    qdot ：关节速度指令

随后统一做工程限幅：
    - 关节速度限幅  |qdot| <= qdot_max
    - 关节限位投影  q_min <= q_target <= q_max
    - 末端指令限幅  |x_target - x_current| <= ee_step_max
    - 数值保护      不用 np.linalg.inv，直接解线性方程组；并检测 NaN/Inf
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DLSConfig:
    """DLS 控制器参数（全部为工程可调项，默认值来自 FR3 实际尺度）。"""

    kp: float = 1.0                 # 位置增益（对单步指令，1.0 表示一步到位）
    damping: float = 0.05           # λ，阻尼系数
    qdot_max: float = 2.0           # rad/s，关节速度上限
    ee_step_max: float = 0.02       # m，单次求解允许的最大末端位移
    max_dq_per_step: float = 0.05   # rad，单步关节位置增量上限
    singular_threshold: float = 1e-6


@dataclass
class DLSResult:
    q_target: np.ndarray
    qdot_cmd: np.ndarray
    error_norm: float
    applied_delta: np.ndarray
    saturated: bool
    notes: list[str] = field(default_factory=list)


class DLSIKController:
    """阻尼最小二乘位置 IK。

    Parameters
    ----------
    q_min, q_max : 关节位置限位（rad）
    """

    def __init__(
        self,
        q_min: np.ndarray,
        q_max: np.ndarray,
        config: DLSConfig | None = None,
    ) -> None:
        self.q_min = np.asarray(q_min, dtype=np.float64).reshape(-1)
        self.q_max = np.asarray(q_max, dtype=np.float64).reshape(-1)
        self.cfg = config or DLSConfig()
        self.n = self.q_min.size

    # ------------------------------------------------------------------
    def clamp_delta(self, delta: np.ndarray) -> tuple[np.ndarray, bool]:
        """把末端位移指令限幅到 ee_step_max（保持方向）。"""
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n > self.cfg.ee_step_max:
            return d * (self.cfg.ee_step_max / n), True
        return d, False

    # ------------------------------------------------------------------
    def solve(
        self,
        q: np.ndarray,
        jacobian_pos: np.ndarray,
        x_current: np.ndarray,
        x_target: np.ndarray,
        dt: float,
    ) -> DLSResult:
        """求解一步 IK。

        Parameters
        ----------
        q            : (n,) 当前关节位置
        jacobian_pos : (3, n) 末端位置 Jacobian（世界系）
        x_current    : (3,) 当前末端位置
        x_target     : (3,) 期望末端位置（本步）
        dt           : 本步控制周期 (s)
        """
        notes: list[str] = []
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        J = np.asarray(jacobian_pos, dtype=np.float64).reshape(3, self.n)
        x_cur = np.asarray(x_current, dtype=np.float64).reshape(3)
        x_tgt = np.asarray(x_target, dtype=np.float64).reshape(3)

        # ① 末端指令限幅
        delta, sat = self.clamp_delta(x_tgt - x_cur)
        if sat:
            notes.append("ee_step_clamped")
        x_cmd = x_cur + delta

        # ② 位置误差
        e = x_cmd - x_cur

        # ③ DLS：qdot = J^T (J J^T + λ²I)^{-1} Kp e
        lam2 = self.cfg.damping ** 2
        A = J @ J.T + lam2 * np.eye(3)
        try:
            y = np.linalg.solve(A, self.cfg.kp * e)
        except np.linalg.LinAlgError:
            notes.append("linalg_fallback_lstsq")
            y = np.linalg.lstsq(A, self.cfg.kp * e, rcond=None)[0]
        qdot = J.T @ y

        # ④ 数值保护
        if not np.all(np.isfinite(qdot)):
            notes.append("qdot_nonfinite_zeroed")
            qdot = np.zeros(self.n)

        # ⑤ 关节速度限幅
        qdot_norm = float(np.linalg.norm(qdot))
        if qdot_norm > 1e-12:
            scale = min(1.0, self.cfg.qdot_max / qdot_norm)
            if scale < 1.0:
                notes.append("qdot_clamped")
            qdot = qdot * scale

        # ⑥ 位置增量限幅（防止单步跳变）
        dq = qdot * dt
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > self.cfg.max_dq_per_step and dq_norm > 1e-12:
            dq = dq * (self.cfg.max_dq_per_step / dq_norm)
            notes.append("dq_clamped")

        # ⑦ 关节限位投影
        q_target = np.clip(q + dq, self.q_min, self.q_max)
        if not np.allclose(q_target, q + dq):
            notes.append("joint_limit_projected")

        return DLSResult(
            q_target=q_target,
            qdot_cmd=qdot,
            error_norm=float(np.linalg.norm(e)),
            applied_delta=delta,
            saturated=sat,
            notes=notes,
        )


# ----------------------------------------------------------------------
# FR3 默认控制器工厂（所有脚本共用，避免各自调参导致口径不一致）
# ----------------------------------------------------------------------
FR3_Q_MIN = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
FR3_Q_MAX = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
FR3_QDOT_MAX = 2.62      # rad/s，取 7 个关节速度上限中的最小值（FR3 最保守约束）
FR3_EE_STEP_MAX = 0.02   # m，单次求解允许的末端最大位移


def default_fr3_config(control_dt: float) -> "DLSConfig":
    """按控制周期给出合理增益。

    关键点（实测踩过的坑）：任务给定公式中的 qdot_cmd 是**关节速度**（rad/s），
    末端速度为 J·qdot ≈ Kp·e（m/s）。因此若取 Kp=1.0，末端速度只有 0.02 m/s，
    30 Hz 下每控制步仅移动 0.0007 m —— 表现为"控制器几乎不动"。
    正确做法是让增益带上 1/s 量纲：  Kp = 1/control_dt （速度增益）
    这样 J·qdot·dt ≈ e，即"想要多少位移就走多少位移"。
    """
    return DLSConfig(
        kp=1.0 / control_dt,          # 1/s，速度增益
        damping=0.05,
        qdot_max=FR3_QDOT_MAX,
        ee_step_max=FR3_EE_STEP_MAX,
        max_dq_per_step=0.05,
    )


def default_fr3_controller(control_dt: float) -> "DLSIKController":
    return DLSIKController(FR3_Q_MIN, FR3_Q_MAX, default_fr3_config(control_dt))


# ----------------------------------------------------------------------
# 数值 Jacobian（当仿真不提供解析 Jacobian 时的后备方案）
# ----------------------------------------------------------------------
def numerical_position_jacobian(fk_fn, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """用中心差分估计位置 Jacobian(3, n)。

    fk_fn(q) 必须返回末端位置 (3,)。要求 fk_fn 是纯运动学计算（不推进物理）。
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = q.size
    J = np.zeros((3, n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        p_plus = np.asarray(fk_fn(q + dq), dtype=np.float64).reshape(3)
        p_minus = np.asarray(fk_fn(q - dq), dtype=np.float64).reshape(3)
        J[:, i] = (p_plus - p_minus) / (2 * eps)
    return J
