"""Sanity-check the ported environment before spending any training time.

Runs the Gymnasium API conformance checker, then a few random episodes, and
reports the terminal `info` outcomes so we can confirm landings/crashes are
exposed correctly.

    python scripts/check_env.py
"""

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)


def main():
    base = gym.make("RocketLander-v1", continuous=True).unwrapped
    check_env(base, skip_render_check=True)
    print("[ok] Gymnasium env_checker passed")

    env = gym.make("RocketLander-v1", continuous=True, max_episode_steps=1000)
    rng = np.random.default_rng(0)
    outcomes = []
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        assert env.observation_space.contains(obs), "initial obs out of bounds"
        done = False
        steps, total_r = 0, 0.0
        term_info = {}
        while not done:
            action = rng.uniform(-1, 1, size=3).astype(np.float32)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            done = terminated or truncated
            if terminated:
                term_info = info
        outcomes.append((steps, round(total_r, 2),
                         term_info.get("landed"), term_info.get("crashed")))
        print(f"ep {ep}: steps={steps:4d} return={total_r:7.2f} "
              f"landed={term_info.get('landed')} crashed={term_info.get('crashed')}")
    env.close()
    print("[ok] random rollouts completed:", outcomes)


if __name__ == "__main__":
    main()
