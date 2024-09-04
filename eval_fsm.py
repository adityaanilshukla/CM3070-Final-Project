import time
import numpy as np
import gym
import psutil  # For monitoring system resources
from fsm_controller import fsm_control_loop  # Import the FSM control loop
import os
from plot_results import plot_results  # Import the plot_results module
import io
import sys

env = gym.make('RocketLander-v0')

def evaluate_fsm(num_episodes=100):
    episodes = []
    max_gimbal_smoothness = []  # List to store the maximum gimbal angles for each episode
    avg_gimbal_smoothness = []  # List to store the average gimbal angles for each episode
    min_gimbal_smoothness = []  # List to store the minimum gimbal angles for each episode
    max_deviations = []  # List to store the maximum deviations for each episode
    avg_deviations = []  # List to store the average deviations for each episode
    min_deviations = []  # List to store the minimum deviations for each episode
    time_taken_to_land = []  # List to store the time taken to land in seconds for each episode
    landing_successes = []  # List to store landing success for each episode

    for episode in range(num_episodes):
        obs = env.reset()
        start_time = time.time()  # Start timing the episode
        process = psutil.Process()  # Start monitoring the process
        landed = False  # Flag to track if the rocket has landed successfully

        # Capture the standard output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Run the FSM control loop and gather flight data for every timestep
        flight_data, done = fsm_control_loop(env, render=False)

        # Capture the output and check for landing success
        output = sys.stdout.getvalue()
        if "LANDED!!!!!!!!!" in output:
            landed = True

        # Restore standard output
        sys.stdout = old_stdout

        # Record landing success
        landing_successes.append(landed)

        # Process flight data to calculate metrics
        episode_gimbal_smoothness = []
        episode_deviations = []

        for timestep_data in flight_data:
            state = timestep_data['state']

            # Monitor gimbal angle (in radians) - NO abs() to capture both positive and negative values
            gimbal_angle = state['gimbal']
            episode_gimbal_smoothness.append(gimbal_angle)

            # Monitor angle deviation (can be positive or negative, so no abs())
            angle_deviation = state['angle']
            episode_deviations.append(angle_deviation)

        # Record the max, min, and average gimbal angles for the episode
        max_gimbal_angle = max(episode_gimbal_smoothness) if episode_gimbal_smoothness else 0
        min_gimbal_angle = min(episode_gimbal_smoothness) if episode_gimbal_smoothness else 0
        avg_gimbal_angle = np.mean(episode_gimbal_smoothness) if episode_gimbal_smoothness else 0
        max_gimbal_smoothness.append(max_gimbal_angle)
        min_gimbal_smoothness.append(min_gimbal_angle)
        avg_gimbal_smoothness.append(avg_gimbal_angle)

        # Record the max, min, and average deviations for the episode
        max_deviation = max(episode_deviations) if episode_deviations else 0
        min_deviation = min(episode_deviations) if episode_deviations else 0
        avg_deviation = np.mean(episode_deviations) if episode_deviations else 0
        max_deviations.append(max_deviation)
        min_deviations.append(min_deviation)
        avg_deviations.append(avg_deviation)

        # Record the time taken to land for the episode
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_to_land.append(time_taken)

        episodes.append(episode)

    # Generate the relevant plots
    plot_results(
        episodes=episodes,
        max_gimbal_smoothness=max_gimbal_smoothness,
        min_gimbal_smoothness=min_gimbal_smoothness,
        avg_gimbal_smoothness=avg_gimbal_smoothness,
        max_deviations=max_deviations,
        avg_deviations=avg_deviations,
        min_deviations=min_deviations,
        time_taken_to_land=time_taken_to_land,
        model_type='FSM',  # Specify the model type as 'FSM'
        landing_successes=landing_successes,
        # Empty lists for unused plots
        max_response_times=[],
        avg_response_times=[],
        min_response_times=[],
        max_throttle_smoothness=[],
        avg_throttle_smoothness=[],
        min_throttle_smoothness=[],
        avg_cpu_usages=[]
    )

# Run the evaluation
evaluate_fsm(100)  # Evaluate over 100 episodes
