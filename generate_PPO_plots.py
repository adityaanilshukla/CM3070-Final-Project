import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import PPO2
from gym.envs.registration import register

# Create the directory if it doesn't exist
os.makedirs('plots/PPO', exist_ok=True)

# Register the RocketLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Adjust the path as needed
    max_episode_steps=2500,
    reward_threshold=0,
)

# Load the trained PPO model
model_path = os.path.join("model", "ppo2_RocketLander-v0_2024-08-02024338_step_18000000.zip")
model = PPO2.load(model_path)

# Create the environment
env = gym.make('RocketLander-v0')
obs = env.reset()

# Initialize lists to store values for plotting
x_vals = []
y_vals = []
angle_vals = []
gimbal_vals = []
throttle_vals = []
vx_vals = []
vy_vals = []

done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    # Extract the relevant values from the observation
    x_vals.append(obs[0])
    y_vals.append(obs[1])
    angle_vals.append(obs[2])
    throttle_vals.append(obs[5])
    gimbal_vals.append(obs[6])
    vx_vals.append(obs[7])
    vy_vals.append(obs[8] - 3)  # Adjust Vy to account for stationary 3 velocity

    # No need to render the environment for plotting
    # env.render()

# Plotting the results
# Plot Rocket Trajectory
plt.figure()
plt.plot(x_vals, y_vals)
plt.title('Rocket Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.savefig('plots/PPO/trajectory.png')

# Plot Rocket Angle Over Time
plt.figure()
plt.plot(range(len(angle_vals)), angle_vals)
plt.title('Rocket Angle Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Angle (Radians)')
plt.grid(True)
plt.savefig('plots/PPO/angle_correction.png')

# Plot Gimbal Angle Adjustments Over Time
plt.figure()
plt.plot(range(len(gimbal_vals)), gimbal_vals)
plt.title('Gimbal Angle Adjustments Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Gimbal Angle (Radians)')
plt.grid(True)
plt.savefig('plots/PPO/gimbal_angle_adjustments.png')

# Plot Throttle Adjustments Over Time
plt.figure()
plt.plot(range(len(throttle_vals)), throttle_vals)
plt.title('Throttle Adjustments Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Throttle')
plt.grid(True)
plt.savefig('plots/PPO/throttle_adjustments.png')

# Plot Horizontal and Vertical Velocities Over Time
plt.figure()
plt.plot(range(len(vx_vals)), vx_vals, label='Vx (Horizontal Velocity)')
plt.plot(range(len(vy_vals)), vy_vals, label='Vy (Vertical Velocity)')
plt.title('Horizontal and Vertical Velocities Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.savefig('plots/PPO/velocities.png')
