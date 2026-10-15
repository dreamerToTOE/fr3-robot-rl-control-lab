"""RL04 —— 可视化与社会对照表生成。

读取真实训练/评估产出，生成:
  results/rl04/comparison/plots/
    01_training_return_vs_steps.png
    02_eval_return_vs_steps.png
    03_angle_error_vs_time.png
    04_angular_velocity_vs_time.png
    05_torque_vs_time.png
    06_comparison_bars.png
    07_action_saturation.png
  results/rl04/comparison/comparison.csv
  results/rl04/comparison/comparison.md
  results/rl04/<run>/plots/{train_return_vs_steps,eval_return_vs_steps}.png

所有曲线均来自真实记录数据，不做任何人工构造。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import rl04_common as C  # noqa: E402

ALGOS = ["ppo", "sac"]
POLICIES = ["random", "ppo", "sac"]
COLORS = {"ppo": "#1f77b4", "sac": "#d62728", "random": "#7f7f7f"}


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_train_curve(algo: str):
    path = C.RESULTS_ROOT / f"{algo}_seed42" / "train_metrics.csv"
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    steps = np.array([float(r["env_step"]) for r in rows])
    rets = np.array([float(r["episode_return"]) for r in rows])
    return steps, rets


def load_eval_vs_steps(algo: str):
    path = C.RESULTS_ROOT / f"{algo}_seed42" / "eval_vs_steps.csv"
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    steps = np.array([float(r["env_step"]) for r in rows])
    mean = np.array([float(r["mean_return"]) for r in rows])
    std = np.array([float(r["std_return"]) for r in rows])
    return steps, mean, std


def load_trace_matrix(algo: str, run_id: str | None = None):
    """返回 (theta, theta_dot, torque, cum_return) 矩阵 shape=(n_episodes, T)，T 可能不一致则按最短对齐。"""
    run_id = run_id or f"{algo}_seed42"
    path = C.RESULTS_ROOT / run_id / "eval_trace.csv"
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    by_ep: dict[int, list[dict]] = {}
    for r in rows:
        by_ep.setdefault(int(r["episode"]), []).append(r)
    eps = sorted(by_ep)
    T = min(len(by_ep[e]) for e in eps)
    theta = np.array([[float(x["theta_rad"]) for x in by_ep[e][:T]] for e in eps])
    thd = np.array([[float(x["theta_dot_rad_s"]) for x in by_ep[e][:T]] for e in eps])
    tq = np.array([[float(x["torque_nm"]) for x in by_ep[e][:T]] for e in eps])
    cum = np.array([[float(x["cumulative_return"]) for x in by_ep[e][:T]] for e in eps])
    return theta, thd, tq, cum


def load_metrics(algo: str):
    path = C.RESULTS_ROOT / f"{algo}_seed42" / "eval_metrics.json"
    if not path.exists():
        return None
    return C.read_json(path)


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot_train_curves(save_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False
    for algo in ALGOS:
        data = load_train_curve(algo)
        if data is None:
            continue
        any_data = True
        steps, rets = data
        ax.plot(steps, rets, alpha=0.25, color=COLORS[algo], lw=1)
        w = max(5, len(rets) // 20)
        sm = smooth(rets, w)
        ax.plot(steps[w - 1:], sm, color=COLORS[algo], lw=2,
                label=f"{algo.upper()} (raw + {w}-ep moving avg)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("RL04 Training Return vs Environment Steps (Pendulum-v1, seed=42)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "01_training_return_vs_steps.png", dpi=150)
    plt.close(fig)
    return any_data


def plot_eval_vs_steps(save_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False
    for algo in ALGOS:
        data = load_eval_vs_steps(algo)
        if data is None:
            continue
        any_data = True
        steps, mean, std = data
        ax.errorbar(steps, mean, yerr=std, marker="o", color=COLORS[algo],
                    label=f"{algo.upper()} (20-ep eval)", capsize=3)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Mean evaluated episode return")
    ax.set_title("RL04 Evaluation Return vs Training Steps (fixed eval seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "02_eval_return_vs_steps.png", dpi=150)
    plt.close(fig)
    return any_data


def _band_plot(save_dir: Path, fname: str, ylabel: str, title: str, key: str, take_abs=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in POLICIES:
        m = load_trace_matrix(algo)
        if m is None:
            continue
        theta, thd, tq, cum = m
        arr = {"theta": theta, "thd": thd, "tq": tq}[key]
        if take_abs:
            arr = np.abs(arr)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        t = np.arange(len(mu))
        ax.plot(t, mu, color=COLORS[algo], lw=2, label=algo.upper())
        ax.fill_between(t, np.maximum(mu - sd, 0), mu + sd, color=COLORS[algo], alpha=0.2)
    ax.set_xlabel("Time step within episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / fname, dpi=150)
    plt.close(fig)


def plot_comparison_bars(save_dir: Path, metrics: dict):
    keys = [
        ("mean_episode_return", "Mean episode return", False),
        ("upright_fraction", "Upright fraction", False),
        ("action_saturation_ratio", "Action saturation ratio", False),
        ("mean_abs_angle_error_rad", "Mean |angle error| (rad)", False),
        ("mean_abs_torque_nm", "Mean |torque| (N·m)", False),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4.2))
    for ax, (k, title, _) in zip(axes, keys):
        vals = [metrics[p].get(k, 0.0) if metrics[p] else 0.0 for p in POLICIES]
        bars = ax.bar([p.upper() for p in POLICIES], vals,
                      color=[COLORS[p] for p in POLICIES])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3g}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("RL04 Policy Comparison — Random vs PPO vs SAC (20-episode deterministic eval)")
    fig.tight_layout()
    fig.savefig(save_dir / "06_comparison_bars.png", dpi=150)
    plt.close(fig)


def plot_saturation(save_dir: Path, metrics: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    vals = [metrics[p].get("action_saturation_ratio", 0.0) if metrics[p] else 0.0 for p in POLICIES]
    bars = ax1.bar([p.upper() for p in POLICIES], vals, color=[COLORS[p] for p in POLICIES])
    ax1.set_title("Action saturation ratio (|u| >= 0.99*2.0 N·m)")
    ax1.set_ylabel("Fraction of steps")
    ax1.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2%}",
                 ha="center", va="bottom", fontsize=9)

    for algo in POLICIES:
        m = load_trace_matrix(algo)
        if m is None:
            continue
        tq = m[2].reshape(-1)
        ax2.hist(tq, bins=40, alpha=0.5, color=COLORS[algo], label=algo.upper(), density=True)
    ax2.set_title("Torque distribution (all eval steps)")
    ax2.set_xlabel("Torque (N·m)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "07_action_saturation.png", dpi=150)
    plt.close(fig)


def _load_run_csv(run_id: str, name: str):
    path = C.RESULTS_ROOT / run_id / name
    if not path.exists():
        return None
    return read_csv_rows(path)


def plot_ppo_budget(save_dir: Path):
    """PPO 预算诊断图：prescribed 50k vs extended 300k。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for run_id, label, color in [
        ("ppo_seed42", "PPO prescribed 50k steps", "#ff7f0e"),
        ("ppo_seed42_300k", "PPO extended 300k steps", "#1f77b4"),
    ]:
        tr = _load_run_csv(run_id, "train_metrics.csv")
        if tr:
            s = np.array([float(r["env_step"]) for r in tr])
            r = np.array([float(r["episode_return"]) for r in tr])
            w = max(5, len(r) // 25)
            ax1.plot(s, r, color=color, alpha=0.18, lw=1)
            ax1.plot(s[w - 1:], smooth(r, w), color=color, lw=2, label=label)
        ev = _load_run_csv(run_id, "eval_vs_steps.csv")
        if ev:
            st = np.array([float(r["env_step"]) for r in ev])
            mn = np.array([float(r["mean_return"]) for r in ev])
            sd = np.array([float(r["std_return"]) for r in ev])
            ax2.errorbar(st, mn, yerr=sd, marker="o", color=color, capsize=3, label=label)

    ax1.axhline(-1258.78, ls="--", color="#7f7f7f", lw=1.2, label="Random baseline (-1258.8)")
    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Episode return")
    ax1.set_title("PPO training return: 50k vs 300k budget")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhline(-1258.78, ls="--", color="#7f7f7f", lw=1.2)
    ax2.set_xlabel("Training environment steps")
    ax2.set_ylabel("Mean evaluated episode return")
    ax2.set_title("PPO evaluation return vs steps (fixed eval seeds)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "08_ppo_budget_diagnosis.png", dpi=150)
    plt.close(fig)


def write_comparison(metrics: dict, out_dir: Path):
    fields = [
        "policy", "mean_episode_return", "std_episode_return", "mean_episode_length",
        "mean_abs_angle_error_rad", "mean_abs_angular_velocity_rad_s", "mean_abs_torque_nm",
        "max_abs_torque_nm", "action_saturation_ratio", "upright_fraction",
    ]
    rows = []
    for p in POLICIES:
        m = metrics.get(p)
        if not m:
            continue
        rows.append({k: m.get(k) for k in fields})
    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# RL04 对照结果 — Random vs PPO vs SAC (Pendulum-v1)",
        "",
        "评估协议：每个策略 20 个独立 episode，固定 seed = 1000..1019，deterministic 动作，"
        "每 episode 固定 200 步 TimeLimit（truncation 不计为失败）。",
        "",
        "容差定义：upright = |theta| ≤ 0.2 rad 且 |theta_dot| ≤ 1.0 rad/s；"
        "saturation = |u| ≥ 0.99 × 2.0 N·m。",
        "",
        "| Policy | Mean Return | Std Return | Mean |theta| (rad) | Mean |theta_dot| (rad/s) | Mean |u| (N·m) | Max |u| (N·m) | Saturation | Upright frac |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy'].upper()} | {r['mean_episode_return']:.2f} | {r['std_episode_return']:.2f} | "
            f"{r['mean_abs_angle_error_rad']:.3f} | {r['mean_abs_angular_velocity_rad_s']:.3f} | "
            f"{r['mean_abs_torque_nm']:.3f} | {r['max_abs_torque_nm']:.3f} | "
            f"{r['action_saturation_ratio']:.2%} | {r['upright_fraction']:.2%} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    comp_dir = C.ensure_dir(C.RESULTS_ROOT / "comparison")
    plots_dir = C.ensure_dir(comp_dir / "plots")

    metrics = {p: load_metrics(p) for p in POLICIES}

    plot_train_curves(plots_dir)
    plot_eval_vs_steps(plots_dir)
    _band_plot(plots_dir, "03_angle_error_vs_time.png", "|theta| (rad)",
               "RL04 |Angle Error| vs Time", "theta")
    _band_plot(plots_dir, "04_angular_velocity_vs_time.png", "|theta_dot| (rad/s)",
               "RL04 |Angular Velocity| vs Time", "thd")
    _band_plot(plots_dir, "05_torque_vs_time.png", "|torque| (N·m)",
               "RL04 |Torque| vs Time", "tq")
    plot_comparison_bars(plots_dir, metrics)
    plot_saturation(plots_dir, metrics)
    plot_ppo_budget(plots_dir)
    write_comparison(metrics, comp_dir)

    # 每个 run 自己的曲线
    for algo in ALGOS:
        run_plots = C.ensure_dir(C.RESULTS_ROOT / f"{algo}_seed42" / "plots")
        data = load_train_curve(algo)
        if data:
            fig, ax = plt.subplots(figsize=(9, 5))
            steps, rets = data
            ax.plot(steps, rets, alpha=0.3, color=COLORS[algo], lw=1, label="raw")
            w = max(5, len(rets) // 20)
            sm = smooth(rets, w)
            ax.plot(steps[w - 1:], sm, color=COLORS[algo], lw=2, label=f"{w}-ep moving avg")
            ax.set_xlabel("Environment steps")
            ax.set_ylabel("Episode return")
            ax.set_title(f"RL04 {algo.upper()} Training Return vs Steps (seed=42)")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_plots / "train_return_vs_steps.png", dpi=150)
            plt.close(fig)
        ev = load_eval_vs_steps(algo)
        if ev:
            fig, ax = plt.subplots(figsize=(9, 5))
            steps, mean, std = ev
            ax.errorbar(steps, mean, yerr=std, marker="o", color=COLORS[algo], capsize=3)
            ax.set_xlabel("Training environment steps")
            ax.set_ylabel("Mean evaluated episode return")
            ax.set_title(f"RL04 {algo.upper()} Evaluation Return vs Steps")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_plots / "eval_return_vs_steps.png", dpi=150)
            plt.close(fig)

    print(f"[plot] figures -> {plots_dir}")
    print(f"[plot] comparison -> {comp_dir / 'comparison.csv'}, {comp_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
