"""Demonstration-seeded SAC (SACfD-style) for RocketLander-v1.

Vanilla SAC lands 0% on the hard task: the +1 landing reward sits ~1000 sim
steps past a region of state space that undirected exploration never reaches, so
the actor never sees a single successful trajectory and has no gradient toward
one. The fix here is classic learning-from-demonstration: pre-load the replay
buffer with successful landings from a working continuous controller
(FastLandingController, the PPO-distilled fast lander, ~84-88% landed), so from
the first gradient step the critic already sees terminal landing transitions and
the actor is pulled toward them.

Two pieces live here:

  * generate_demos(): roll the fast controller over hard-difficulty episodes and
    store the successful (and partial) transitions as an .npz. CRITICAL obs-space
    detail: the controller DECODES raw observations, but SAC TRAINS on statically
    normalized observations. So the env is run with static_normalize=False (raw
    obs for the controller), and we store normalized = (raw - MEAN) / STD, using
    the env's own STATIC_OBS_MEAN / STATIC_OBS_STD. The formula is verified
    against a lockstep static_normalize=True env before generation.

  * DemoMixReplayBuffer: a ReplayBuffer subclass that holds a second, frozen demo
    buffer and overrides sample() to return a mix -- demo_ratio of every batch
    drawn from the demos, the rest from the online buffer. Demos live in a
    RESERVED partition, not the 300k online ring, so they are never evicted.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch as th
from torch.nn import functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update

import rocketrl.envs  # noqa: F401  (registers RocketLander-v1)
from rocketrl.configs import EnvConfig
from rocketrl.controllers.lqr import make_fast_controller
from rocketrl.envs.rocket_lander import (
    STATIC_OBS_MEAN, STATIC_OBS_STD, START_SPEED, H, SHIP_HEIGHT)


# Demo-generation seed base. Kept disjoint from evaluate.py's EVAL_SEED_OFFSET
# (1_000_000) and the fast controller's tuning bank (2_000_000..) so the
# demonstrations never overlap the held-out evaluation seeds.
DEMO_SEED_BASE = 3_000_000

# The keys stored in / loaded from the demo .npz.
_DEMO_KEYS = ("observations", "next_observations", "actions",
              "rewards", "dones", "timeouts")


def _normalize(raw_obs: np.ndarray) -> np.ndarray:
    """Apply the env's static normalization to a raw observation.

    Identical to what RocketLander.step does when static_normalize=True, so the
    stored demo observations match exactly what SAC sees at train time.
    """
    normed = (raw_obs - STATIC_OBS_MEAN[:len(raw_obs)]) / STATIC_OBS_STD[:len(raw_obs)]
    return normed.astype(np.float32)


def _verify_normalization(env_cfg: EnvConfig, seed: int = DEMO_SEED_BASE,
                          n_steps: int = 30, tol: float = 1e-5) -> float:
    """Confirm _normalize(raw) matches a lockstep static_normalize=True env.

    Builds a raw env and a statically-normalized env with the same seed, drives
    both with the identical fast-controller actions (computed from the raw obs),
    and asserts the manually normalized raw obs equals the static env's obs.
    Returns the max absolute difference seen.
    """
    raw_env = gym.make(env_cfg.env_id, continuous=True,
                       max_episode_steps=env_cfg.max_episode_steps,
                       static_normalize=False,
                       init_x_range=env_cfg.init_x_range,
                       init_vx_random=env_cfg.init_vx_random,
                       init_spin=env_cfg.init_spin)
    norm_env = gym.make(env_cfg.env_id, continuous=True,
                        max_episode_steps=env_cfg.max_episode_steps,
                        static_normalize=True,
                        init_x_range=env_cfg.init_x_range,
                        init_vx_random=env_cfg.init_vx_random,
                        init_spin=env_cfg.init_spin)
    controller = make_fast_controller()
    controller.reset()
    raw_obs, _ = raw_env.reset(seed=seed)
    norm_obs, _ = norm_env.reset(seed=seed)
    max_diff = 0.0
    for _ in range(n_steps):
        action = np.asarray(controller.act(raw_obs), dtype=np.float32)
        manual = _normalize(raw_obs)
        max_diff = max(max_diff, float(np.abs(manual - norm_obs).max()))
        raw_obs, _, term, trunc, _ = raw_env.step(action)
        norm_obs, _, _, _, _ = norm_env.step(action)
        if term or trunc:
            break
    raw_env.close()
    norm_env.close()
    if max_diff > tol:
        raise AssertionError(
            f"normalization mismatch: max|manual - static_env| = {max_diff} > {tol}")
    return max_diff


def generate_demos(n_episodes: int = 100,
                   seed_base: int = DEMO_SEED_BASE,
                   env_cfg: EnvConfig | None = None,
                   keep_timeouts: bool = True,
                   out_path: str | None = None,
                   verbose: bool = True) -> dict[str, Any]:
    """Roll the fast controller over hard-difficulty episodes, store transitions.

    Keeps landed episodes (the point) and, if keep_timeouts, partial (truncated,
    non-crash) episodes; drops crashes (a crash terminal would teach the wrong
    terminal sign). Stores statically-normalized observations with SB3-correct
    done / timeout flags. If out_path is given, saves an .npz there.

    Returns a dict with the arrays (under _DEMO_KEYS) and a 'stats' sub-dict.
    """
    if env_cfg is None:
        env_cfg = EnvConfig()   # hard defaults: x 0.3, vx 0.4, spin 1.4, 3000 steps

    max_diff = _verify_normalization(env_cfg, seed=seed_base)
    if verbose:
        print(f"[demos] normalization verified (max|manual-static| = {max_diff:.2e})")

    env = gym.make(env_cfg.env_id, continuous=True,
                   max_episode_steps=env_cfg.max_episode_steps,
                   static_normalize=False,
                   init_x_range=env_cfg.init_x_range,
                   init_vx_random=env_cfg.init_vx_random,
                   init_spin=env_cfg.init_spin)
    controller = make_fast_controller()

    obs_buf: list[np.ndarray] = []
    next_obs_buf: list[np.ndarray] = []
    action_buf: list[np.ndarray] = []
    reward_buf: list[float] = []
    done_buf: list[bool] = []
    timeout_buf: list[bool] = []

    n_landed = 0
    n_crashed = 0
    n_timeout = 0
    n_kept = 0

    try:
        for ep in range(n_episodes):
            seed = seed_base + ep
            controller.reset()
            raw_obs, info = env.reset(seed=seed)

            ep_obs: list[np.ndarray] = []
            ep_next: list[np.ndarray] = []
            ep_act: list[np.ndarray] = []
            ep_rew: list[float] = []
            ep_done: list[bool] = []
            ep_timeout: list[bool] = []

            while True:
                action = np.asarray(controller.act(raw_obs), dtype=np.float32)
                next_raw, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                is_timeout = bool(truncated and not terminated)

                ep_obs.append(_normalize(raw_obs))
                ep_next.append(_normalize(next_raw))
                ep_act.append(action)
                ep_rew.append(float(reward))
                ep_done.append(bool(done))
                ep_timeout.append(is_timeout)

                raw_obs = next_raw
                if done:
                    break

            landed = bool(info.get("landed", False))
            crashed = bool(info.get("crashed", False))
            timeout = bool(truncated and not terminated)
            if landed:
                n_landed += 1
            elif crashed:
                n_crashed += 1
            elif timeout:
                n_timeout += 1

            # keep landed (always) and partial/timeout (optional); drop crashes
            keep = landed or (timeout and keep_timeouts)
            if keep:
                n_kept += 1
                obs_buf.extend(ep_obs)
                next_obs_buf.extend(ep_next)
                action_buf.extend(ep_act)
                reward_buf.extend(ep_rew)
                done_buf.extend(ep_done)
                timeout_buf.extend(ep_timeout)

            if verbose and (ep + 1) % 10 == 0:
                print(f"[demos] episode {ep + 1}/{n_episodes} "
                      f"landed={n_landed} crashed={n_crashed} timeout={n_timeout} "
                      f"kept_transitions={len(obs_buf)}")
    finally:
        env.close()

    demos = {
        "observations": np.asarray(obs_buf, dtype=np.float32),
        "next_observations": np.asarray(next_obs_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.float32),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "timeouts": np.asarray(timeout_buf, dtype=np.float32),
    }
    stats = {
        "n_episodes": n_episodes,
        "n_landed": n_landed,
        "n_crashed": n_crashed,
        "n_timeout": n_timeout,
        "n_kept_episodes": n_kept,
        "n_transitions": int(demos["observations"].shape[0]),
        "landing_fraction": n_landed / max(n_episodes, 1),
        # count terminal landings by the env flag, NOT by reward: the landed
        # terminal reward is base 1.0 plus a clipped negative terminal penalty,
        # so it lands around 0.5-1.0 and is rarely exactly +1.
        "n_landing_terminals": int(((demos["dones"] > 0.5)
                                    & (demos["timeouts"] < 0.5)
                                    & (demos["rewards"] > 0.3)).sum()),
    }
    demos["stats"] = stats

    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez_compressed(out_path, **{k: demos[k] for k in _DEMO_KEYS},
                            stats=np.array(str(stats)))
        if verbose:
            print(f"[demos] saved {stats['n_transitions']} transitions -> {out_path}")
    if verbose:
        print(f"[demos] stats: {stats}")
    return demos


def load_demos(demo_path: str) -> dict[str, np.ndarray]:
    """Load the demo arrays from an .npz saved by generate_demos."""
    with np.load(demo_path, allow_pickle=True) as data:
        return {k: data[k] for k in _DEMO_KEYS}


# ---------------------------------------------------------------------------
# Potential-based descent-profile shaping.
#
# The BC runs showed demo-induced landings APPEAR (dose-responsive to bc_coef)
# but then ERODE: the gamma-0.997 crash-attractor pulls the actor back off the
# demo manifold. BC fights that at the action level; it never reshapes the value
# landscape. The env's own telescoping potential already covers distance/speed/
# angle, so the one descent dimension it MISSES is vertical-speed tracking of the
# known-good braking profile. Phi rewards vy matching v_ref(h); as a POTENTIAL
# (Ng/Harada/Russell, F = gamma*Phi(s') - Phi(s)) it is policy-invariant -- the
# per-episode shaping sum collapses to -Phi(s0), a start-state constant, so it
# cannot prefer a crash trajectory over a landing one; it only densifies the
# gradient toward the descent corridor.
# ---------------------------------------------------------------------------
# Fast-controller braking curve (coordinator's spec for the shaping target; V_TD
# is deliberately 3.0 here, slightly gentler than the controller's own 5.0).
V_TD_SHAPE = 3.0
K_TERM_SHAPE = 0.33
A_BRAKE_SHAPE = 9.0
# Altitude span mapped by obs[1] (mirrors the env's y encoding; reimplemented
# here rather than imported from controllers/lqr.py to avoid coupling to a file
# under concurrent edit).
_SHIPHEIGHT = H / 20.0 + SHIP_HEIGHT
_ALT_SPAN = H - _SHIPHEIGHT


def _descent_v_ref(h_m: float) -> float:
    """Known-good target descent speed (m/s, negative) at altitude h_m."""
    h = max(h_m, 0.0)
    return -min(V_TD_SHAPE + K_TERM_SHAPE * h,
                float(np.sqrt(V_TD_SHAPE ** 2 + 2.0 * A_BRAKE_SHAPE * h)))


class DescentProfileShapingWrapper(gym.Wrapper):
    """Add a potential-based descent-velocity-tracking reward to each step.

    Phi(s) = -w_shape * |vy - v_ref(h)|, and the shaping reward is
    gamma*Phi(s') - Phi(s). Terminal (terminated) states are absorbing, so
    Phi=0 there per the Ng convention; truncation (timeout) is NOT absorbing, so
    its Phi is the real potential and the bootstrap term is kept. This keeps the
    per-episode shaping sum at the policy-invariant -Phi(s0) and leaks no
    spurious terminal bonus.

    Decodes h and vy from the observation the same way the controllers do. When
    the underlying env is static_normalize=True the wrapper sees NORMALIZED obs,
    so it first inverts the static normalization before decoding SI units.
    """

    def __init__(self, env, w_shape: float = 0.1, gamma: float = 0.997) -> None:
        super().__init__(env)
        self.w_shape = float(w_shape)
        self.gamma = float(gamma)
        self._static_norm = bool(getattr(env.unwrapped, "static_normalize", False))
        self._prev_phi = 0.0

    def _phi(self, obs: np.ndarray) -> float:
        raw = np.asarray(obs, dtype=np.float32)
        if self._static_norm:
            n = len(raw)
            raw = raw * STATIC_OBS_STD[:n] + STATIC_OBS_MEAN[:n]
        h_m = (float(raw[1]) / 2.0 + 0.5) * _ALT_SPAN
        vy_m = float(raw[8]) * START_SPEED
        return -self.w_shape * abs(vy_m - _descent_v_ref(h_m))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = 0.0 if terminated else self._phi(obs)   # absorbing terminal -> 0
        shaping = self.gamma * phi_next - self._prev_phi
        self._prev_phi = phi_next
        info["shaping_reward"] = float(shaping)
        return obs, float(reward) + float(shaping), terminated, truncated, info


class DemoMixReplayBuffer(ReplayBuffer):
    """ReplayBuffer that mixes a frozen demo partition into every sampled batch.

    The online transitions go into the usual ring buffer (inherited). A second,
    frozen ReplayBuffer holds the demonstrations and is never written to after
    construction, so demos are never evicted. sample() returns demo_ratio of each
    batch from the demo buffer and the remainder from the online buffer.

    Constructed by SB3 via replay_buffer_class + replay_buffer_kwargs; SB3 passes
    (buffer_size, observation_space, action_space, device, n_envs,
    optimize_memory_usage) plus our kwargs (demo_path / demos, demo_ratio).
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: th.device | str = "auto",
                 n_envs: int = 1,
                 optimize_memory_usage: bool = False,
                 handle_timeout_termination: bool = True,
                 demo_path: str | None = None,
                 demos: dict[str, np.ndarray] | None = None,
                 demo_ratio: float = 0.25) -> None:
        super().__init__(
            buffer_size, observation_space, action_space,
            device=device, n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.demo_ratio = float(demo_ratio)

        if demos is None:
            if demo_path is None:
                raise ValueError("DemoMixReplayBuffer needs demo_path or demos")
            demos = load_demos(demo_path)

        n_demo = int(demos["observations"].shape[0])
        if n_demo == 0:
            raise ValueError("demo set is empty; generate demos first")

        # Reserved, frozen demo partition. n_envs=1 (demos are a flat stream);
        # buffer_size == n_demo so nothing is ever evicted; same device as the
        # online buffer so the two sub-samples concatenate without a device move.
        self.demo_buffer = ReplayBuffer(
            n_demo, observation_space, action_space,
            device=device, n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
        )
        # Fill the demo buffer's arrays directly (vectorized; the per-env axis is
        # length 1). This matches exactly what repeated .add() calls would write.
        self.demo_buffer.observations[:, 0, :] = demos["observations"]
        self.demo_buffer.next_observations[:, 0, :] = demos["next_observations"]
        self.demo_buffer.actions[:, 0, :] = demos["actions"].reshape(n_demo, -1)
        self.demo_buffer.rewards[:, 0] = demos["rewards"]
        self.demo_buffer.dones[:, 0] = demos["dones"]
        self.demo_buffer.timeouts[:, 0] = demos["timeouts"]
        self.demo_buffer.full = True
        self.demo_buffer.pos = 0
        self.n_demo = n_demo
        # Number of leading demo rows in the most recent sampled batch. SACfD.train
        # reads this to apply the behavior-cloning loss to exactly those rows.
        self._last_n_demo = 0

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        """Return a demo_ratio / (1 - demo_ratio) mix of demo and online samples.

        Demo rows are placed FIRST (indices 0 : n_demo) so a SACfD trainer can
        slice them out of the batch for the behavior-cloning auxiliary loss.
        """
        n_demo = int(round(self.demo_ratio * batch_size))
        n_demo = min(n_demo, batch_size)
        n_online = batch_size - n_demo
        self._last_n_demo = n_demo

        if n_demo == 0:
            return super().sample(batch_size, env=env)
        if n_online == 0:
            return self.demo_buffer.sample(batch_size, env=env)

        demo = self.demo_buffer.sample(n_demo, env=env)
        online = super().sample(n_online, env=env)
        # Concatenate only the five real tensor fields, demos first. The sixth
        # ReplayBufferSamples field (discounts) is None for a plain ReplayBuffer,
        # so passing it through th.cat would throw; set it to None explicitly.
        return ReplayBufferSamples(
            observations=th.cat([demo.observations, online.observations], dim=0),
            actions=th.cat([demo.actions, online.actions], dim=0),
            next_observations=th.cat([demo.next_observations, online.next_observations], dim=0),
            dones=th.cat([demo.dones, online.dones], dim=0),
            rewards=th.cat([demo.rewards, online.rewards], dim=0),
            discounts=None,
        )


class SACfD(SAC):
    """SAC with a behavior-cloning auxiliary loss on the demonstration rows.

    Plain demo-buffer seeding (DemoMixReplayBuffer alone) left the critic to
    bootstrap Q at (demo_state, policy_action) pairs the online policy never
    visits; that extrapolation error compounded and Q diverged to the thousands.
    The standard SACfD/DDPGfD cure is to ANCHOR the actor to the demonstrated
    actions: on the demo rows of each mixed batch, add bc_coef * MSE(policy mean
    action, demo action) to the actor loss. The actor then stops querying wild
    out-of-distribution actions, which both improves behavior and tames the
    critic.

    Relies on DemoMixReplayBuffer placing the demo rows first in each sampled
    batch (indices 0 : replay_buffer._last_n_demo).
    """

    def __init__(self, *args, bc_coef: float = 0.5,
                 bc_coef_final: float | None = None,
                 bc_q_filter: bool = False, **kwargs) -> None:
        # Stored before super().__init__ so train() (never called during init)
        # always sees them; SAC itself has no such attributes to clobber.
        # bc_coef is the INITIAL weight; if bc_coef_final is set, the weight
        # decays linearly from bc_coef to bc_coef_final over training progress
        # (strong demo pull during discovery, then RL fine-tunes).
        self.bc_coef = float(bc_coef)
        self.bc_coef_final = None if bc_coef_final is None else float(bc_coef_final)
        self.bc_q_filter = bool(bc_q_filter)
        super().__init__(*args, **kwargs)

    def _current_bc_coef(self) -> float:
        """BC weight for the current step (constant, or the linear schedule)."""
        if self.bc_coef_final is None:
            return self.bc_coef
        total = max(int(getattr(self, "_total_timesteps", 0)), 1)
        progress_remaining = max(0.0, 1.0 - self.num_timesteps / total)
        return self.bc_coef_final + (self.bc_coef - self.bc_coef_final) * progress_remaining

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # This is SB3 2.9.2 SAC.train() with a behavior-cloning term spliced into
        # the actor loss (marked BC below). The rest is kept verbatim so the SAC
        # update semantics are unchanged.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses = [], [], []
        bc_coef_now = self._current_bc_coef()

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # Leading rows of the batch are demonstrations (DemoMixReplayBuffer).
            n_demo = int(getattr(self.replay_buffer, "_last_n_demo", 0))
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # --- BC auxiliary loss (SACfD) ------------------------------------
            # On the demo rows only, pull the actor's deterministic (squashed
            # mean) action toward the demonstrated action. The action space is
            # exactly [-1, 1]^3, so the squashed mean and the stored demo action
            # live on the same scale -- no (un)scaling needed.
            bc_loss_value = 0.0
            if bc_coef_now > 0.0 and n_demo > 0:
                demo_obs = replay_data.observations[:n_demo]
                demo_actions = replay_data.actions[:n_demo]
                bc_mean_actions = self.actor(demo_obs, deterministic=True)
                if self.bc_q_filter:
                    # Q-filter: only clone where the demo action is judged at
                    # least as good as the policy's current action by the critic.
                    with th.no_grad():
                        q_demo = th.min(th.cat(self.critic(demo_obs, demo_actions), dim=1),
                                        dim=1, keepdim=True)[0]
                        q_pi = th.min(th.cat(self.critic(demo_obs, bc_mean_actions.detach()), dim=1),
                                      dim=1, keepdim=True)[0]
                        keep = (q_demo > q_pi).float()
                    per_row = ((bc_mean_actions - demo_actions) ** 2).mean(dim=1, keepdim=True)
                    bc_loss = (keep * per_row).sum() / (keep.sum() + 1e-8)
                else:
                    bc_loss = F.mse_loss(bc_mean_actions, demo_actions)
                actor_loss = actor_loss + bc_coef_now * bc_loss
                bc_loss_value = float(bc_loss.item())
            bc_losses.append(bc_loss_value)
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/bc_coef", bc_coef_now)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


__all__ = [
    "DEMO_SEED_BASE",
    "generate_demos",
    "load_demos",
    "DemoMixReplayBuffer",
    "DescentProfileShapingWrapper",
    "SACfD",
]
