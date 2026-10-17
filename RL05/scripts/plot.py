#!/usr/bin/env python3
"""RL05 —— 训练曲线与对照图（全部来自真实仿真/评估数据）。

输出到 RL05/results/rl05/comparison/plots/:
    01_training_return_vs_steps.png      训练回报曲线
    02_eval_success_rate_vs_steps.png    评估成功率 vs 步数（逐 checkpoint，若有）
    03_final_position_error.png          最终位置误差（含各策略分布）
    04_episode_length.png                回合步数
    05_action_magnitude_saturation.png   动作幅值与饱和率
    06_ee_error_vs_time.png              末端位置误差随时间
    07_policy_comparison.png             Random / Classical / SAC / PPO 对照
以及 comparison/comparison.csv 与 comparison.md。

用法:
    python3 <RL05>/scripts/plot.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RL05 = Path(__file__).resolve().parents[1]
RESULTS = RL05 / "results" / "rl05"
COMP = RESULTS / "comparison"
PLOTS = COMP / "plots"

POLICIES = ["random", "classical", "oracle", "sac", "ppo"]
LABELS = {"random": "Random", "classical": "Classical DLS", "oracle": "Oracle (myopic)",
          "sac": "SAC", "ppo": "PPO"}
COLORS = {"random": "#7f7f7f", "classical": "#2ca02c", "oracle": "#9467bd",
          "sac": "#d62728", "ppo": "#1f77b4"}


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_train(algo: str, seed: int = 42):
    # 优先正式 run（v2 奖励重平衡版），否则回退到 v1 试点
    for run in (f"{algo}_seed{seed}_v2", f"{algo}_seed{seed}"):
        p = RESULTS / run / "train_metrics.csv"
        if not p.exists():
            continue
        rows = read_csv(p)
        return {
            "run": run,
            "steps": np.array([float(r["env_step"]) for r in rows]),
            "ret": np.array([float(r["episode_return"]) for r in rows]),
            "len": np.array([float(r["episode_length"]) for r in rows]),
            "succ": np.array([float(r["is_success"]) for r in rows]),
            "dist": np.array([float(r["final_distance"]) for r in rows]),
        }
    return None


def smooth(x, w):
    if len(x) < w or w < 2:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    metrics = {p: load_json(RESULTS / f"{p}_eval" / "eval_metrics.json") for p in POLICIES}
    traces = {}
    for p in POLICIES:
        tp = RESULTS / f"{p}_eval" / "eval_trace.csv"
        if tp.exists():
            rows = read_csv(tp)
            by_ep: dict[int, list[dict]] = {}
            for r in rows:
                by_ep.setdefault(int(r["episode"]), []).append(r)
            eps = sorted(by_ep)
            T = min(len(by_ep[e]) for e in eps)
            traces[p] = np.array([[float(x["dist_m"]) for x in by_ep[e][:T]] for e in eps])

    # 01 训练回报
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ("sac", "ppo"):
        d = load_train(algo)
        if not d:
            continue
        ax.plot(d["steps"], d["ret"], color=COLORS[algo], alpha=0.2, lw=1)
        w = max(5, len(d["ret"]) // 25)
        ax.plot(d["steps"][w - 1:], smooth(d["ret"], w), color=COLORS[algo], lw=2,
                label=f"{LABELS[algo]} (raw + {w}-ep moving avg)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("RL05 Training Return vs Environment Steps (FR3 Cartesian Reach)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "01_training_return_vs_steps.png", dpi=150); plt.close(fig)

    # 02 评估成功率 vs 步数
    fig, ax = plt.subplots(figsize=(9, 5))
    any_ = False
    for algo in ("sac", "ppo"):
        p = RESULTS / f"{algo}_sweep" / "eval_vs_steps.csv"
        if p.exists():
            rows = read_csv(p)
            st = [float(r["env_step"]) for r in rows]
            sr = [float(r["success_rate"]) for r in rows]
            ax.plot(st, sr, marker="o", color=COLORS[algo], label=LABELS[algo])
            any_ = True
        d = load_train(algo)
        if d is not None:  # 训练期滚动成功率（辅助曲线）
            w = max(10, len(d["succ"]) // 20)
            ax.plot(d["steps"][w - 1:], smooth(d["succ"], w), ls="--", color=COLORS[algo],
                    alpha=0.5, label=f"{LABELS[algo]} train roll. success")
            any_ = True
    for p in ("random", "classical"):
        m = metrics.get(p)
        if m:
            ax.axhline(m["success_rate"], ls=":", color=COLORS[p],
                       label=f"{LABELS[p]} eval = {m['success_rate']:.0%}")
            any_ = True
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("RL05 Evaluation Success Rate vs Steps")
    if any_:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "02_eval_success_rate_vs_steps.png", dpi=150); plt.close(fig)

    # 03 最终位置误差
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    vals = [metrics[p]["mean_final_position_error_m"] for p in POLICIES if metrics.get(p)]
    names = [LABELS[p] for p in POLICIES if metrics.get(p)]
    cols = [COLORS[p] for p in POLICIES if metrics.get(p)]
    bars = ax1.bar(names, vals, color=cols)
    ax1.axhline(0.02, ls="--", color="k", lw=1, label="success tol = 0.02 m")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Mean final position error (m)")
    ax1.set_title("Mean final position error")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")
    for p in POLICIES:
        ep = RESULTS / f"{p}_eval" / "eval_episodes.csv"
        if ep.exists():
            fe = [float(r["final_error"]) for r in read_csv(ep)]
            ax2.hist(fe, bins=25, alpha=0.5, color=COLORS[p], label=LABELS[p], density=True)
    ax2.axvline(0.02, ls="--", color="k", lw=1)
    ax2.set_xlabel("Final position error (m)")
    ax2.set_ylabel("Density")
    ax2.set_title("Final error distribution")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "03_final_position_error.png", dpi=150); plt.close(fig)

    # 04 回合步数
    fig, ax = plt.subplots(figsize=(9, 4.6))
    steps = [metrics[p]["mean_episode_length"] for p in POLICIES if metrics.get(p)]
    bars = ax.bar(names, steps, color=cols)
    for b, v in zip(bars, steps):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean episode length (control steps)")
    ax.set_title("Mean episode length (max = 100 control steps)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOTS / "04_episode_length.png", dpi=150); plt.close(fig)

    # 05 动作幅值与饱和
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    mags = [metrics[p]["mean_action_norm"] for p in POLICIES if metrics.get(p)]
    sats = [metrics[p]["action_saturation_ratio"] for p in POLICIES if metrics.get(p)]
    b1 = ax1.bar(names, mags, color=cols)
    for b, v in zip(b1, mags):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Mean ||action||")
    ax1.set_title("Action magnitude")
    ax1.grid(alpha=0.3, axis="y")
    b2 = ax2.bar(names, sats, color=cols)
    for b, v in zip(b2, sats):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Saturation ratio")
    ax2.set_title("Action saturation (|u| at bound)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOTS / "05_action_magnitude_saturation.png", dpi=150); plt.close(fig)

    # 06 末端误差 vs 时间
    fig, ax = plt.subplots(figsize=(9, 5))
    for p, arr in traces.items():
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        t = np.arange(len(mu)) / (1.0 / 120.0 * 4)
        ax.plot(t, mu, color=COLORS[p], lw=2, label=LABELS[p])
        ax.fill_between(t, np.maximum(mu - sd, 0), mu + sd, color=COLORS[p], alpha=0.18)
    ax.axhline(0.02, ls="--", color="k", lw=1, label="success tol 0.02 m")
    ax.set_xlabel("Simulation time within episode (s)")
    ax.set_ylabel("|ee - target| (m)")
    ax.set_title("RL05 End-effector position error vs time (mean ± std over eval targets)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "06_ee_error_vs_time.png", dpi=150); plt.close(fig)

    # 07 综合对照
    keys = [("success_rate", "Success rate", False),
            ("mean_final_position_error_m", "Mean final error (m)", False),
            ("mean_return", "Mean return", False),
            ("action_smoothness_rms", "Action smoothness RMS", False)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 4.3))
    for ax, (k, title, _) in zip(axes, keys):
        v = [metrics[p].get(k, 0.0) for p in POLICIES if metrics.get(p)]
        b = ax.bar(names, v, color=cols)
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv:.3g}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("RL05 Random vs Classical DLS vs SAC vs PPO (100 fixed reachable targets)")
    fig.tight_layout(); fig.savefig(PLOTS / "07_policy_comparison.png", dpi=150); plt.close(fig)

    # comparison.csv / md
    fields = ["policy", "n_episodes", "success_rate", "mean_final_position_error_m",
              "median_final_position_error_m", "mean_time_to_success_s", "mean_return",
              "mean_episode_length", "action_saturation_ratio", "mean_action_norm",
              "action_smoothness_rms", "total_joint_limit_violations", "total_collision_steps"]
    rows = []
    for p in POLICIES:
        m = metrics.get(p)
        if not m:
            continue
        rows.append({k: m.get(k) for k in fields})
    if rows:
        with open(COMP / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

        lines = ["# RL05 对照结果 — Random / Classical DLS / SAC / PPO（FR3 Cartesian Reach）", "",
                 "评估协议：同一组 100 个**已验证可达**目标；关节精确复位到 READY；30 Hz 控制；",
                 "成功判据 `|ee - target| <= 0.02 m` 连续保持 3 个控制步；上限 100 控制步；评估期不更新参数。", "",
                 "| Policy | Success | Mean final err (m) | Median err (m) | Time-to-success (s) | Mean return | Ep len | Sat ratio | Smoothness RMS | Limit viol | Collision steps |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            tts = r["mean_time_to_success_s"]
            lines.append(
                f"| {LABELS[r['policy']]} | {r['success_rate']:.1%} | {r['mean_final_position_error_m']:.4f} | "
                f"{r['median_final_position_error_m']:.4f} | "
                f"{(f'{tts:.3f}' if tts is not None else '—')} | {r['mean_return']:.2f} | "
                f"{r['mean_episode_length']:.1f} | {r['action_saturation_ratio']:.1%} | "
                f"{r['action_smoothness_rms']:.4f} | {r['total_joint_limit_violations']} | "
                f"{r['total_collision_steps']} |")
        (COMP / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[plot] -> {PLOTS}")
    for p in POLICIES:
        m = metrics.get(p)
        if m:
            print(f"  {LABELS[p]:14s} success={m['success_rate']:.1%} "
                  f"err={m['mean_final_position_error_m']:.4f} return={m['mean_return']:.2f}")
        else:
            print(f"  {LABELS[p]:14s} (no eval_metrics.json yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
