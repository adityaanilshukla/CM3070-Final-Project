import time
import numpy as np
import psutil  # For monitoring system resources
import gc  # Garbage collection
import matplotlib.pyplot as plt
from fsm_controller import *  # Import the FSM control loop
import os

# Ensure the output directory exists
output_dir = 'plots/single-episode/FSM'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def evaluate_fsm_single_episode():
    # Reinitialize the environment for a single episode
    global env
    env.close()  # Close the existing environment
    del env  # Delete it to free memory
    env = gym.make('RocketLander-v0')  # Recreate the environment

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

    # Run the FSM control loop and gather flight data and actions
    flight_data, actions, done = fsm_control_loop(env, render=False)

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
    plt.title('Rocket Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.savefig(f'{output_dir}/trajectory.png')

    plt.figure()
    plt.plot(range(len(angle_vals)), angle_vals)
    plt.title('Rocket Angle Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Angle (Radians)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/angle_correction.png')

    plt.figure()
    plt.plot(range(len(throttle_vals)), throttle_vals)
    plt.title('Throttle Adjustment Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Throttle')
    plt.grid(True)
    plt.savefig(f'{output_dir}/throttle_adjustments.png')

    plt.figure()
    plt.plot(range(len(gimbal_vals)), gimbal_vals)
    plt.title('Gimbal Angle Adjustments Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Gimbal Angle (Radians)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/gimbal_angle_adjustments.png')

    plt.figure()
    plt.plot(range(len(vx_vals)), vx_vals, label='Vx (Horizontal Velocity)')
    plt.plot(range(len(vy_vals)), vy_vals, label='Vy (Vertical Velocity)')
    plt.title('Horizontal and Vertical Velocities Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/velocities.png')

# Run the evaluation for a single episode
evaluate_fsm_single_episode()
