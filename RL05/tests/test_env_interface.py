#!/usr/bin/env python3
"""RL05 —— FR3 Reach 环境接口测试（任务书第六阶段要求的 10 项必测）。

未全部通过则不得进入 PPO/SAC 正式训练。

用法:
    cd /home/ubuntu2004/isaacsim-4.5.0
    PYTHONPATH=<RL05>/.pydeps ./python.sh <RL05>/tests/test_env_interface.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RL05 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL05 / "envs"))
sys.path.insert(0, str(RL05 / "controllers"))

OUT = RL05 / "results" / "rl05" / "env_interface_tests.json"
READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])
Q_MIN = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
Q_MAX = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])


class Tests:
    def __init__(self):
        self.results: list[dict] = []

    def check(self, name: str, ok: bool, detail: dict | str | None = None) -> bool:
        self.results.append({"test": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}", flush=True)
        return bool(ok)

    @property
    def all_pass(self) -> bool:
        return all(r["pass"] for r in self.results)


def main() -> int:
    rep: dict = {"utc": datetime.now(timezone.utc).isoformat(), "success": False}
    t = Tests()
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                             "anti_aliasing": 0, "multi_gpu": False,
                             "sync_loads": False, "fast_shutdown": True})

        from fr3_reach_env import FR3ReachEnv, FR3ReachSim, ReachConfig

        cfg = ReachConfig(reset_joint_noise=0.0)
        sim = FR3ReachSim(cfg=cfg)
        rep["ee_index"] = sim.ee_idx
        rep["link_names_head"] = sim.link_names[:24]

        train_path = RL05 / "assets" / "targets_train.json"
        if not train_path.is_file():
            raise FileNotFoundError("run scripts/generate_targets.py first")
        pool = np.asarray(json.loads(train_path.read_text())["targets"], dtype=np.float64)

        env = FR3ReachEnv(sim, cfg, target_pool=pool, seed=42)
        obs, info = env.reset(seed=42)

        # --- 1. reset 后回到合法初始状态 ---
        q, qd = sim.get_joint_state()
        ee = sim.get_ee_pos()
        ok1 = (q.shape == (7,) and np.all(np.isfinite(q)) and np.all(np.isfinite(qd))
               and np.all(q >= Q_MIN - 1e-2) and np.all(q <= Q_MAX + 1e-2)
               and np.all(np.abs(qd) < 0.5) and np.linalg.norm(ee - READY_EE) < 0.05)
        t.check("1_reset_legal_initial_state", ok1, {
            "q_dev_from_ready_max": float(np.max(np.abs(q - READY))),
            "qdot_norm": float(np.linalg.norm(qd)),
            "ee_dist_from_home": float(np.linalg.norm(ee - READY_EE)),
        })

        # --- 2. observation 维度与有限性 ---
        ok2 = (obs.shape == (20,) and obs.dtype == np.float32
               and np.all(np.isfinite(obs)) and env.observation_space.shape == (20,))
        t.check("2_observation_shape_finite", ok2, {"shape": list(obs.shape), "dtype": str(obs.dtype)})

        # --- 3+4. 动作缩放与方向 ---
        env.reset(seed=1)
        ee0 = sim.get_ee_pos()
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        ee1 = sim.get_ee_pos()
        dpos = ee1 - ee0
        cos_x = float(dpos[0] / max(np.linalg.norm(dpos), 1e-12))
        ok3 = float(np.linalg.norm(dpos)) <= cfg.action_scale * 1.6
        ok4 = cos_x > 0.7
        t.check("3_action_scale_bounded", ok3, {
            "measured_step_m": float(np.linalg.norm(dpos)), "action_scale": cfg.action_scale})
        t.check("4_positive_x_action_moves_ee_plus_x", ok4, {
            "dpos": dpos.tolist(), "cos_with_plus_x": cos_x})

        # --- 5. Jacobian 与实际运动一致 ---
        env.reset(seed=2)
        q_a, _ = sim.get_joint_state()
        ee_a = sim.get_ee_pos()
        J = sim.get_position_jacobian()
        dq = np.zeros(7); dq[0] = 0.05
        sim.art.set_joint_position_targets(q_a + dq)
        sim.step_physics(120)
        q_b, _ = sim.get_joint_state()
        ee_b = sim.get_ee_pos()
        dq_actual = q_b - q_a
        measured = ee_b - ee_a
        predicted = J @ dq_actual
        cos = float(np.dot(measured, predicted) / max(np.linalg.norm(measured) * np.linalg.norm(predicted), 1e-12))
        ratio = float(np.linalg.norm(measured) / max(np.linalg.norm(predicted), 1e-12))
        t.check("5_jacobian_matches_motion", cos > 0.95 and 0.7 < ratio < 1.4, {
            "cos_sim": cos, "magnitude_ratio": ratio,
            "measured": measured.tolist(), "predicted": predicted.tolist()})

        # --- 6. reward 分项符合设计意图 ---
        env.reset(seed=3)
        target = env.current_target.copy()
        terms_ok = {}
        _, r1, _, _, i1 = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        terms_ok["smoothness_zero_on_first_step"] = abs(i1["reward_terms"]["smoothness"]) < 1e-12
        terms_ok["action_zero_when_action_zero"] = abs(i1["reward_terms"]["action"]) < 1e-12
        terms_ok["no_joint_limit_in_nominal"] = abs(i1["reward_terms"]["joint_limit"]) < 1e-12
        # 朝目标方向走 → progress > 0
        ee = sim.get_ee_pos()
        dirv = target - ee
        dirv = dirv / max(np.linalg.norm(dirv), 1e-9)
        env.reset(seed=4)
        _, _, _, _, ia = env.step((dirv * 1.0).astype(np.float32))
        env.reset(seed=4)
        _, _, _, _, ib = env.step((-dirv * 1.0).astype(np.float32))
        terms_ok["progress_positive_toward_target"] = ia["reward_terms"]["progress"] > 0
        terms_ok["progress_negative_away_from_target"] = ib["reward_terms"]["progress"] < 0
        # smoothness 在动作变化时为负
        env.reset(seed=5)
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        _, _, _, _, i2 = env.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        terms_ok["smoothness_negative_on_change"] = i2["reward_terms"]["smoothness"] < 0
        # action penalty 随 ||a||^2 增长
        env.reset(seed=6)
        _, _, _, _, i3 = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3))
        terms_ok["action_penalty_negative"] = i3["reward_terms"]["action"] < 0
        t.check("6_reward_terms_match_design", all(terms_ok.values()), terms_ok)

        # --- 7. 成功条件能触发 ---
        env.reset(seed=7)
        ee = sim.get_ee_pos()
        env.set_target(ee.copy())          # 目标就是当前位置
        success = False
        for _ in range(cfg.hold_steps + 2):
            _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            if info["success"]:
                success = True
            if term or trunc:
                break
        t.check("7_success_triggers", success and info["success"], {
            "hold_steps": cfg.hold_steps, "dist": info["distance"]})

        # --- 8. 时间限制正确 truncate ---
        env.reset(seed=8)
        trunc_seen = False
        steps = 0
        for _ in range(cfg.max_control_steps + 5):
            _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            steps += 1
            if trunc:
                trunc_seen = True
                break
            if term:
                break
        t.check("8_truncation_at_limit", trunc_seen and steps == cfg.max_control_steps, {
            "steps": steps, "max_control_steps": cfg.max_control_steps})

        # --- 9. 连续多次 reset 无漂移/状态污染 ---
        drifts = []
        for i in range(5):
            o, _ = env.reset(seed=100 + i)
            q_r, qd_r = sim.get_joint_state()
            ee_r = sim.get_ee_pos()
            drifts.append({
                "obs_finite": bool(np.all(np.isfinite(o))),
                "ee_dist_from_home": float(np.linalg.norm(ee_r - READY_EE)),
                "qdot_norm": float(np.linalg.norm(qd_r)),
            })
        ok9 = all(d["obs_finite"] and d["ee_dist_from_home"] < 0.05 and d["qdot_norm"] < 0.5 for d in drifts)
        t.check("9_repeated_reset_no_drift", ok9, {"drifts": drifts})

        # --- 10. 随机策略完整 episode ---
        rng = np.random.default_rng(0)
        eps = []
        crashed = False
        try:
            for k in range(3):
                env.reset(seed=200 + k)
                tot = 0.0
                steps = 0
                while True:
                    _, r, term, trunc, info = env.step(rng.uniform(-1, 1, 3).astype(np.float32))
                    tot += r
                    steps += 1
                    if term or trunc:
                        break
                eps.append({"return": float(tot), "steps": steps, "success": info["success"]})
        except Exception as exc:
            crashed = True
            eps.append({"error": f"{type(exc).__name__}: {exc}"})
        t.check("10_random_policy_full_episode", (not crashed) and len(eps) == 3, {"episodes": eps})

        # --- 额外：随机策略成功率与经典基线对照（参考，不作为门禁） ---
        rep["all_pass"] = t.all_pass
        rep["tests"] = t.results
        rep["success"] = t.all_pass
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print("ENVTEST_ALL_PASS" if t.all_pass else "ENVTEST_HAS_FAILURES", flush=True)
        return 0 if t.all_pass else 1
    except Exception as exc:
        rep["error"] = str(exc)
        rep["traceback"] = traceback.format_exc()
        rep["tests"] = t.results
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"ENVTEST_ERROR {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        import os
        os._exit(0 if 't' in dir() and t.all_pass else 1)


# READY 位姿下的末端位置（由 build 报告得到，用于复位漂移判定）
READY_EE = np.array([0.30725347995758057, 9.201467037200928e-05, 0.6965761780738831])


if __name__ == "__main__":
    sys.exit(main())
