import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
import os
from ppo_controller import *  # Import the PPO control loop

# Ensure the output directory exists
output_dir = 'plots/single-episode/PPO'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def evaluate_ppo_single_episode():
    obs = env.reset()
    process = psutil.Process()

    # Flight data lists
    x_vals = []
    y_vals = []
    angle_vals = []
    throttle_vals = []
    gimbal_vals = []
    vx_vals = []
    vy_vals = []

    # Run the PPO control loop and gather flight data and actions
    flight_data, actions, done = run_ppo_model(env, model, render=False)

    for timestep_data in flight_data:
        state = timestep_data['state']

        # Collect data for plots
        x_vals.append(state['x'])
        y_vals.append(state['y'])
        angle_vals.append(state['angle'])
        throttle_vals.append(state['throttle'])
        gimbal_vals.append(state['gimbal'])
        vx_vals.append(state['vx'])
        vy_vals.append(state['vy'])

    # Create the plots
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.title('Rocket Trajectory (PPO)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.savefig(f'{output_dir}/trajectory.png')

    plt.figure()
    plt.plot(range(len(angle_vals)), angle_vals)
    plt.title('Rocket Angle Over Time (PPO)')
    plt.xlabel('Time Steps')
    plt.ylabel('Angle (Radians)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/angle_correction.png')

    plt.figure()
    plt.plot(range(len(throttle_vals)), throttle_vals)
    plt.title('Throttle Adjustment Over Time (PPO)')
    plt.xlabel('Time Steps')
    plt.ylabel('Throttle')
    plt.grid(True)
    plt.savefig(f'{output_dir}/throttle_adjustments.png')

    plt.figure()
    plt.plot(range(len(gimbal_vals)), gimbal_vals)
    plt.title('Gimbal Angle Adjustments Over Time (PPO)')
    plt.xlabel('Time Steps')
    plt.ylabel('Gimbal Angle (Radians)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/gimbal_angle_adjustments.png')

    plt.figure()
    plt.plot(range(len(vx_vals)), vx_vals, label='Vx (Horizontal Velocity)')
    plt.plot(range(len(vy_vals)), vy_vals, label='Vy (Vertical Velocity)')
    plt.title('Horizontal and Vertical Velocities Over Time (PPO)')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/velocities.png')

# Run the evaluation for a single episode
evaluate_ppo_single_episode()
