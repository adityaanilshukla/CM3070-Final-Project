from gymnasium.envs.registration import register

from rocketrl.envs.rocket_lander import RocketLander

register(
    id="RocketLander-v1",
    entry_point="rocketrl.envs.rocket_lander:RocketLander",
    max_episode_steps=3000,
)

__all__ = ["RocketLander"]
