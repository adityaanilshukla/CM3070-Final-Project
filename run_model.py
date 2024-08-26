import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from gym.envs.registration import register

# Register the RocketLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Adjust the path as needed
    max_episode_steps=4000,
    reward_threshold=0,
)

# Function to create the environment
def make_env():
    env = gym.make('RocketLander-v0')
    return env

# Load the trained model
# model_path = "model/ppo2_RocketLander-v0_2024-08-02 02:43:38.zip"
model_path = "model/ppo2_RocketLander-v0_2024-08-02 02:43:38_step_18000000.zip"
model = PPO2.load(model_path)

# Create the environment
env = DummyVecEnv([make_env])

# Run the model in the environment
obs = env.reset()
done = False
step = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if step%100==0:
        print("x: {:.2f}, y: {:.2f}, angle: {:.2f}, leg1_contact: {}, leg2_contact: {}, throttle: {:.2f}, gimbal: {:.2f}, vx: {:.2f}, vy: {:.2f}, v_angle: {:.2f}".format(*obs[0]))
    step += 1
env.close()
