"""RL04 —— 可选：录制 PPO / SAC 控制演示视频。

在无图形界面环境下使用 SDL_VIDEODRIVER=dummy + rgb_array 渲染，
若渲染不可用则优雅跳过（返回码 0），绝不阻塞训练/评估主流程。

用法:
    python scripts/record_video.py --algo ppo --episodes 1 --seed 1000
    python scripts/record_video.py --algo sac --episodes 1 --seed 1000

产物: results/rl04/<algo>_seed42/videos/demo_ep<seed>.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402

import rl04_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "sac", "random"])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--run-id", default=None, help="默认 <algo>_seed42")
    args = ap.parse_args()

    run_id = args.run_id or f"{args.algo}_seed42"
    run_dir = C.RESULTS_ROOT / run_id
    vid_dir = C.ensure_dir(run_dir / "videos")

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[video:{args.algo}] SKIP: imageio unavailable ({exc})")
        return 0

    try:
        env = C.make_env(C.ENV_ID, render_mode="rgb_array")
    except Exception as exc:
        print(f"[video:{args.algo}] SKIP: cannot create render env ({exc})")
        return 0

    model = None
    if args.algo != "random":
        from stable_baselines3 import PPO, SAC

        cls = PPO if args.algo == "ppo" else SAC
        model_path = run_dir / "final_model.zip"
        if not model_path.exists():
            print(f"[video:{args.algo}] SKIP: {model_path} missing")
            return 0
        model = cls.load(str(model_path), device="cpu")

    for ep in range(args.episodes):
        seed = args.seed + ep
        try:
            obs, _ = env.reset(seed=seed)
            frame = env.render()
        except Exception as exc:
            print(f"[video:{args.algo}] SKIP: render failed ({exc})")
            env.close()
            return 0

        frames = [np.asarray(frame)]
        done = False
        while not done:
            if args.algo == "random":
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            try:
                frames.append(np.asarray(env.render()))
            except Exception:
                pass
            done = terminated or truncated

        out = vid_dir / f"demo_ep{seed}.mp4"
        try:
            imageio.mimsave(str(out), frames, fps=30)
            print(f"[video:{args.algo}] saved {out} ({len(frames)} frames)")
        except Exception as exc:
            print(f"[video:{args.algo}] SKIP: encode failed ({exc})")
            env.close()
            return 0

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
