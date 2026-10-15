"""RL04 —— 多随机种子汇总（seed 42/43/44）。

读取每个 (algo, seed) 的 eval_metrics.json（20-episode 独立评估），
计算跨种子 mean ± std 与逐策略明细，输出:
    results/rl04/comparison/multiseed_summary.csv
    results/rl04/comparison/multiseed_summary.md
    results/rl04/comparison/plots/09_multiseed_return.png
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

SEEDS = [42, 43, 44]
METRIC_KEYS = [
    "mean_episode_return", "std_episode_return", "mean_abs_angle_error_rad",
    "mean_abs_torque_nm", "action_saturation_ratio", "upright_fraction",
]


def collect(algo: str, seeds: list[int]) -> dict[int, dict | None]:
    out = {}
    for s in seeds:
        p = C.RESULTS_ROOT / f"{algo}_seed{s}" / "eval_metrics.json"
        out[s] = C.read_json(p) if p.exists() else None
    return out


def main() -> int:
    comp = C.ensure_dir(C.RESULTS_ROOT / "comparison")
    plots = C.ensure_dir(comp / "plots")

    rows = []
    per_algo: dict[str, dict] = {}
    for algo in ("ppo", "sac"):
        data = collect(algo, SEEDS)
        per_algo[algo] = data
        for s, m in data.items():
            if m is None:
                continue
            row = {"algo": algo, "seed": s}
            row.update({k: round(m.get(k, float("nan")), 4) for k in METRIC_KEYS})
            rows.append(row)

    if not rows:
        print("[multiseed] no eval_metrics.json found")
        return 0

    fields = ["algo", "seed"] + METRIC_KEYS
    with open(comp / "multiseed_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# RL04 多随机种子汇总（Pendulum-v1, 50k steps, 20-episode 评估）",
        "",
        "| Algo | Seed | Mean Return | Std | Mean \\|theta\\| (rad) | Mean \\|u\\| (N·m) | Saturation | Upright frac |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['algo'].upper()} | {r['seed']} | {r['mean_episode_return']:.1f} | "
            f"{r['std_episode_return']:.1f} | {r['mean_abs_angle_error_rad']:.3f} | "
            f"{r['mean_abs_torque_nm']:.3f} | {r['action_saturation_ratio']:.2%} | "
            f"{r['upright_fraction']:.2%} |"
        )
    lines += ["", "## 跨种子统计", ""]
    summary = {}
    for algo in ("ppo", "sac"):
        ms = [m for m in per_algo[algo].values() if m]
        if not ms:
            continue
        rets = np.array([m["mean_episode_return"] for m in ms])
        up = np.array([m["upright_fraction"] for m in ms])
        summary[algo] = (rets.mean(), rets.std(ddof=1) if len(rets) > 1 else 0.0,
                         up.mean(), up.std(ddof=1) if len(up) > 1 else 0.0, len(ms))
        lines.append(
            f"- **{algo.upper()}** (n={len(ms)} seeds): mean return = {rets.mean():.1f} "
            f"± {(_std(rets)):.1f}, upright fraction = {up.mean():.2%} ± {(_std(up)):.2%}"
        )
    (comp / "multiseed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 图
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, algo in enumerate(("ppo", "sac")):
        ms = [m for m in per_algo[algo].values() if m]
        if not ms:
            continue
        vals = [m["mean_episode_return"] for m in ms]
        seeds = [s for s, m in per_algo[algo].items() if m]
        ax.bar(np.arange(len(vals)) + i * 0.38, vals, width=0.36,
               label=algo.upper(), color="#1f77b4" if algo == "ppo" else "#d62728")
        for j, (v, s) in enumerate(zip(vals, seeds)):
            ax.text(j + i * 0.38, v, f"s{s}", ha="center", va="top", fontsize=8)
    ax.axhline(-1258.78, ls="--", color="#7f7f7f", lw=1.2, label="Random (-1258.8)")
    ax.set_ylabel("Mean evaluated episode return (20 ep)")
    ax.set_title("RL04 multi-seed comparison (PPO vs SAC, 50k steps)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots / "09_multiseed_return.png", dpi=150)
    plt.close(fig)

    print(f"[multiseed] -> {comp / 'multiseed_summary.csv'}")
    for algo, (m, s, um, us, n) in summary.items():
        print(f"  {algo.upper()}: return {m:.1f} ± {s:.1f} (n={n}), upright {um:.2%}")
    return 0


def _std(x: np.ndarray) -> float:
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
