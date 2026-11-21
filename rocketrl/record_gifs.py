"""Record one landing-attempt GIF per controller, for the README showcase.

Every controller is run for exactly ONE episode on the SAME fixed seed (so the
GIFs are visually comparable: same start position, same wind-free initial
condition), through the shared `Controller` protocol (see
rocketrl/controllers/__init__.py) so this script never duplicates per-
controller logic -- it only drives whatever controller object it is handed
through env.reset() / controller.act() / env.step(), exactly like
rocketrl/evaluate.py's run_episode.

The raw renderer is 504x720 at 60 FPS (rocketrl/envs/rocket_lander.py), which
is far too large/heavy for a GitHub-hosted GIF. To keep each file under ~5 MB
this script: (1) keeps only every Nth rendered frame, derived from
--target-fps (default 24, so ~60/24 -> every 2nd-3rd frame), and (2) resizes
each kept frame by --scale (default 0.45) before encoding.

Usage:
    uv run python -m rocketrl.record_gifs --classical --random
    uv run python -m rocketrl.record_gifs --sac-run runs/sac_s0 --ppo-run runs/ppo_s0
    uv run python -m rocketrl.record_gifs --classical --random --seed 7 --out-dir assets/gifs
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image

import gymnasium as gym

import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)
from rocketrl.controllers import Controller, RandomController
from rocketrl.envs.rocket_lander import FPS

# GIF color depth: the scene is a handful of flat-shaded polygons on a solid
# background, so an adaptive 128-color palette loses no visible detail while
# keeping the encoded file small.
GIF_PALETTE_COLORS = 128


def build_requested_controllers(
    record_classical: bool,
    record_random: bool,
    sac_run_dir: Optional[str],
    ppo_run_dir: Optional[str],
    seed: int,
) -> list[tuple[str, Controller]]:
    """Build the (file_tag, controller) pairs the CLI asked for.

    file_tag is the fixed output basename ("sac", "ppo", "classical",
    "random"), independent of the controller's own display name, so e.g. an
    SAC run directory named "runs/sac_s0" still writes to sac.gif.
    """
    requested: list[tuple[str, Controller]] = []

    if record_classical:
        from rocketrl.controllers.lqr import make_classical_controller

        requested.append(("classical", make_classical_controller()))

    if record_random:
        requested.append(("random", RandomController(seed=seed)))

    if sac_run_dir is not None:
        from rocketrl.controllers.rl import load_rl_controller

        requested.append(("sac", load_rl_controller(sac_run_dir, algo="sac", name="sac")))

    if ppo_run_dir is not None:
        from rocketrl.controllers.rl import load_rl_controller

        requested.append(("ppo", load_rl_controller(ppo_run_dir, algo="ppo", name="ppo")))

    return requested


def record_episode_frames(
    env: gym.Env, controller: Controller, seed: int, max_steps: int
) -> tuple[list[np.ndarray], dict]:
    """Run one fixed-seed episode and return (rgb_frames, final_info).

    Mirrors rocketrl/evaluate.py's run_episode, but collects a render()
    frame after the reset and after every step instead of computing metrics.
    """
    controller.reset()
    observation, info = env.reset(seed=seed)
    frames = [env.render()]

    terminated = False
    truncated = False
    step_count = 0
    while not (terminated or truncated) and step_count < max_steps:
        action = np.asarray(controller.act(observation), dtype=np.float32)
        observation, _reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        step_count += 1

    return frames, info


def frames_to_gif(
    frames: list[np.ndarray], output_path: str, target_fps: float, scale: float
) -> None:
    """Downsample, resize, and encode rendered RGB frames as a looping GIF."""
    frame_skip = max(1, round(FPS / target_fps))
    actual_fps = FPS / frame_skip
    frame_duration_ms = int(round(1000.0 / actual_fps))

    kept_frames = frames[::frame_skip]

    pil_frames: list[Image.Image] = []
    for raw_frame in kept_frames:
        image = Image.fromarray(raw_frame)
        if scale != 1.0:
            new_size = (int(round(image.width * scale)), int(round(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        # Quantize to a small adaptive palette: keeps the GIF well under the
        # 5 MB GitHub-friendly target without a visible quality loss on this
        # flat-shaded scene.
        image = image.convert("P", palette=Image.ADAPTIVE, colors=GIF_PALETTE_COLORS)
        pil_frames.append(image)

    first_frame = pil_frames[0]
    remaining_frames = pil_frames[1:]
    first_frame.save(
        output_path,
        save_all=True,
        append_images=remaining_frames,
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record one fixed-seed landing-attempt GIF per controller."
    )
    parser.add_argument("--seed", type=int, default=0, help="fixed env/controller seed shared by every recorded controller")
    parser.add_argument("--out-dir", type=str, default="assets/gifs", help="directory to write <controller>.gif into")
    parser.add_argument("--max-steps", type=int, default=1000, help="max_episode_steps for the env, and the hard cap on recorded frames")
    parser.add_argument("--target-fps", type=float, default=24.0, help="approximate GIF playback FPS (frames are skipped down from the 60 FPS render)")
    parser.add_argument("--scale", type=float, default=0.45, help="resize factor applied to each 504x720 rendered frame")
    parser.add_argument("--classical", action="store_true", help="record the classical (cascaded-PID) controller -> classical.gif")
    parser.add_argument("--random", action="store_true", help="record the random baseline controller -> random.gif")
    parser.add_argument("--sac-run", type=str, default=None, help="run_dir of a trained SAC model -> sac.gif")
    parser.add_argument("--ppo-run", type=str, default=None, help="run_dir of a trained PPO model -> ppo.gif")
    args = parser.parse_args()

    requested_controllers = build_requested_controllers(
        record_classical=args.classical,
        record_random=args.random,
        sac_run_dir=args.sac_run,
        ppo_run_dir=args.ppo_run,
        seed=args.seed,
    )
    if not requested_controllers:
        raise SystemExit(
            "no controllers selected; pass one or more of "
            "--classical / --random / --sac-run <dir> / --ppo-run <dir>"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    env = gym.make(
        "RocketLander-v1",
        continuous=True,
        max_episode_steps=args.max_steps,
        render_mode="rgb_array",
    )
    try:
        for file_tag, controller in requested_controllers:
            frames, final_info = record_episode_frames(env, controller, args.seed, args.max_steps)
            output_path = os.path.join(args.out_dir, f"{file_tag}.gif")
            frames_to_gif(frames, output_path, args.target_fps, args.scale)

            file_size_mb = os.path.getsize(output_path) / 1e6
            landed = bool(final_info.get("landed", False))
            crashed = bool(final_info.get("crashed", False))
            outcome = "landed" if landed else ("crashed" if crashed else "timed out")
            print(
                f"[{file_tag}] {outcome} after {len(frames) - 1} steps -> "
                f"{output_path} ({file_size_mb:.2f} MB, {len(frames)} raw frames)"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
