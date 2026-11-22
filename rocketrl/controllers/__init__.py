"""Controller abstraction shared by every agent in the comparison.

A Controller maps a RAW environment observation to an action in the env's
action space. This lets the evaluation harness treat the learned policies (SAC,
PPO) and the classical controller (LQR / PID) through one uniform interface, so
all of them run on the identical seeded episodes.

Raw observation layout (what env.step returns, 10-D, continuous mode):
    obs[0] = 2 * x_distance            horizontal offset from pad (x; +right), ~[-1,1]
    obs[1] = 2 * (y_distance - 0.5)    vertical offset above pad (y), ~[-1,1]
    obs[2] = angle                     body angle, folded to ~[-1,1] (0 = upright)
    obs[3] = left leg ground contact   {0, 1}
    obs[4] = right leg ground contact  {0, 1}
    obs[5] = 2 * (throttle - 0.5)      current main-engine throttle, [-1,1]
    obs[6] = gimbal / GIMBAL_THRESHOLD current engine gimbal, [-1,1]
    obs[7] = vx / START_SPEED          horizontal velocity (scaled)
    obs[8] = vy / START_SPEED          vertical velocity (scaled, negative = descending)
    obs[9] = angular velocity          body spin rate (raw)

Continuous action layout (Box(3), each in [-1,1]):
    action[0] = gimbal rate            + swings engine one way, - the other
    action[1] = throttle rate          + increases main throttle, - decreases
    action[2] = side thruster          > 0.5 fire one side, < -0.5 the other, else off
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Controller(Protocol):
    name: str

    def reset(self) -> None:
        """Clear any per-episode internal state (integrators, etc.)."""
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map one raw observation to a continuous action in [-1, 1]^3."""
        ...


class RandomController:
    """Trivial baseline / harness smoke-test control."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1, 1, size=3).astype(np.float32)
