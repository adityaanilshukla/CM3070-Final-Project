"""Pipeline sanity check: do our PPO hyperparameters learn a KNOWN-easy task?

If PPO with the exact rocketrl PPOConfig hyperparameters learns LunarLander
(a standard Box2D task that PPO reliably solves), then the training pipeline is
sound and RocketLander's failure is task-difficulty / budget / env-specific.
If it fails here too, the hyperparameters themselves (e.g. n_epochs=10, lr=3e-4)
are the problem.

    uv run python scripts/sanity_lunarlander.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from rocketrl.configs import PPOConfig

A = PPOConfig()  # the SAME hyperparameters we use for RocketLander

env = make_vec_env("LunarLander-v3", n_envs=16, seed=0)
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

model = PPO(
    "MlpPolicy", env, seed=0, verbose=0, device="cpu",
    learning_rate=A.learning_rate, n_steps=A.n_steps, batch_size=A.batch_size,
    n_epochs=A.n_epochs, gamma=A.gamma, gae_lambda=A.gae_lambda,
    clip_range=A.clip_range, ent_coef=A.ent_coef,
    policy_kwargs=dict(net_arch=A.net_arch),
)

print(f"hyperparams: lr={A.learning_rate} n_steps={A.n_steps} batch={A.batch_size} "
      f"n_epochs={A.n_epochs} ent_coef={A.ent_coef}")
print("LunarLander-v3 is ~solved at +200; random ~ -180. Watching mean reward:")
for chunk in range(6):
    model.learn(total_timesteps=150_000, reset_num_timesteps=False, progress_bar=False)
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    print(f"  steps={ (chunk+1)*150_000:>8d}   eval_mean_reward = {mean_r:7.1f} +/- {std_r:.1f}")
env.close()
