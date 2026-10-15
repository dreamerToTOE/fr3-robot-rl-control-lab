"""RL04 Pendulum 连续控制实验 —— 公共工具模块。

本模块集中定义：
- 路径常量与结果目录约定
- 环境工厂（Pendulum-v1, 统一 observation/action 语义）
- 配置加载（YAML）与配置持久化（JSON）
- 评估指标计算（与算法无关，训练/评估共用）
- 训练指标回调（把 episode return 按 environment step 写入 CSV）

约定（与仓库 AGENTS.md / EXPERIMENT_PROTOCOL.md 对齐）：
- Environment: Pendulum-v1
  - observation: [cos(theta), sin(theta), theta_dot]，shape=(3,)，float32
  - action: 连续标量 torque u ∈ [-2.0, 2.0]（N·m），shape=(1,)
  - reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2)
  - termination: 无提前终止；固定 200 步后 TimeLimit truncation
- 评估指标定义在 eval 中显式记录容差，避免"改标准造成功"。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
RL04_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = RL04_ROOT / "configs"
RESULTS_ROOT = RL04_ROOT / "results" / "rl04"
REF_REPO = RL04_ROOT / ".repo_ref"

ENV_ID = "Pendulum-v1"

# Pendulum-v1 物理常量（与 gymnasium 源码一致，用于饱和/容差判定）
MAX_TORQUE = 2.0
MAX_SPEED = 8.0

# 评估容差（显式记录，评估口径固定，不允许为造成功而修改）
UPRIGHT_ANGLE_TOL = 0.2      # rad, ~11.46 deg
UPRIGHT_ANGULAR_VEL_TOL = 1.0  # rad/s
SATURATION_RATIO_THRESHOLD = 0.99  # |u| >= 0.99*MAX_TORQUE 记为饱和


# ---------------------------------------------------------------------------
# 环境工厂
# ---------------------------------------------------------------------------
def make_env(env_id: str = ENV_ID, seed: int | None = None, render_mode: str | None = None):
    """创建环境。训练时 seed 只在外部通过 reset(seed=) 固定，这里保持一致。"""
    import gymnasium as gym

    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except Exception:
            pass
    return env


# ---------------------------------------------------------------------------
# 观测 -> 物理量反解
# ---------------------------------------------------------------------------
def obs_to_theta(obs: np.ndarray) -> float:
    """[cos, sin, theta_dot] -> theta (rad)，范围 (-pi, pi]。theta=0 表示竖直向上。"""
    c, s = float(obs[0]), float(obs[1])
    return math.atan2(s, c)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------
@dataclass
class EpisodeRecord:
    episode: int
    seed: int
    return_: float
    length: int
    mean_abs_theta: float
    mean_abs_theta_dot: float
    mean_abs_torque: float
    max_abs_torque: float
    saturation_ratio: float
    upright_fraction: float
    terminated: bool
    truncated: bool
    theta_trace: list[float] = field(default_factory=list)
    theta_dot_trace: list[float] = field(default_factory=list)
    torque_trace: list[float] = field(default_factory=list)
    return_trace: list[float] = field(default_factory=list)


def summarise_episodes(records: list[EpisodeRecord], meta: dict | None = None) -> dict:
    """把多个 episode 汇总成 eval_metrics.json 结构。"""
    if not records:
        raise ValueError("no episodes to summarise")

    def _arr(name: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in records], dtype=np.float64)

    rets = _arr("return_")
    out = {
        "num_episodes": len(records),
        "mean_episode_return": float(rets.mean()),
        "std_episode_return": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "min_episode_return": float(rets.min()),
        "max_episode_return": float(rets.max()),
        "mean_episode_length": float(_arr("length").mean()),
        "std_episode_length": float(_arr("length").std(ddof=1)) if len(records) > 1 else 0.0,
        "mean_abs_angle_error_rad": float(_arr("mean_abs_theta").mean()),
        "std_abs_angle_error_rad": float(_arr("mean_abs_theta").std(ddof=1)) if len(records) > 1 else 0.0,
        "mean_abs_angular_velocity_rad_s": float(_arr("mean_abs_theta_dot").mean()),
        "mean_abs_torque_nm": float(_arr("mean_abs_torque").mean()),
        "max_abs_torque_nm": float(_arr("max_abs_torque").max()),
        "action_saturation_ratio": float(_arr("saturation_ratio").mean()),
        "upright_fraction": float(_arr("upright_fraction").mean()),
        "upright_fraction_std": float(_arr("upright_fraction").std(ddof=1)) if len(records) > 1 else 0.0,
        "terminated_episodes": int(sum(r.terminated for r in records)),
        "truncated_episodes": int(sum(r.truncated for r in records)),
        "evaluation_tolerances": {
            "upright_angle_tol_rad": UPRIGHT_ANGLE_TOL,
            "upright_angular_vel_tol_rad_s": UPRIGHT_ANGULAR_VEL_TOL,
            "saturation_threshold_abs_torque_nm": SATURATION_RATIO_THRESHOLD * MAX_TORQUE,
        },
    }
    if meta:
        out.update(meta)
    return out


def records_to_trace_rows(records: list[EpisodeRecord]) -> list[dict]:
    """展开成逐时间步的 eval_trace.csv 行。"""
    rows = []
    for r in records:
        for t, (th, thd, tau, ret) in enumerate(
            zip(r.theta_trace, r.theta_dot_trace, r.torque_trace, r.return_trace)
        ):
            rows.append(
                {
                    "episode": r.episode,
                    "seed": r.seed,
                    "step": t,
                    "theta_rad": th,
                    "abs_theta_rad": abs(th),
                    "theta_dot_rad_s": thd,
                    "abs_theta_dot_rad_s": abs(thd),
                    "torque_nm": tau,
                    "abs_torque_nm": abs(tau),
                    "cumulative_return": ret,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 训练回调：按 environment step 记录 episode return
# ---------------------------------------------------------------------------
try:  # SB3 未安装时仍可 import 本模块（用于绘图等）
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeMetricsCallback(BaseCallback):
        """在每次 episode 结束（done）时记录 return / length / step。"""

        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.rows: list[dict] = []
            self._ep_return = 0.0
            self._ep_len = 0

        def _on_step(self) -> bool:
            reward = float(self.locals.get("rewards", [0.0])[0])
            self._ep_return += reward
            self._ep_len += 1
            for done in self.locals.get("dones", [False]):
                if done:
                    self.rows.append(
                        {
                            "env_step": int(self.num_timesteps),
                            "episode_return": round(self._ep_return, 6),
                            "episode_length": int(self._ep_len),
                            "wall_time_s": round(float(self.model.num_timesteps), 6),
                        }
                    )
                    self._ep_return = 0.0
                    self._ep_len = 0
            return True

except Exception:  # pragma: no cover
    EpisodeMetricsCallback = None  # type: ignore


# ---------------------------------------------------------------------------
# 环境版本记录
# ---------------------------------------------------------------------------
def collect_env_versions() -> dict:
    import platform
    import sys

    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for name in ("gymnasium", "stable_baselines3", "torch", "numpy", "matplotlib", "pandas"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            versions[name] = f"unavailable ({exc})"
    try:
        import torch

        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["torch_num_threads"] = int(torch.get_num_threads())
    except Exception:
        pass
    return versions


def git_commit_of(path: Path) -> str:
    """读取指定仓库目录的 commit（本地副本可能无 .git）。"""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown (no .git)"
    except Exception as exc:
        return f"unknown ({exc})"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def os_env_summary() -> dict:
    return {
        "cwd": os.getcwd(),
        "cpu_count": os.cpu_count(),
    }
