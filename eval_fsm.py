import time
import numpy as np
import gym
import psutil  # For monitoring system resources
from fsm_controller import fsm_control_loop  # Import the FSM control loop
from gym.envs.registration import register
import os
from plot_results import plot_results  # Import the plot_results module

env = gym.make('RocketLander-v0')

def evaluate_fsm_stability_response_time_smoothness_and_time(num_episodes=100):
    episodes = []
    max_deviations = []  # List to store the maximum deviations for each episode
    avg_deviations = []  # List to store the average deviations for each episode
    max_response_times = []  # List to store the maximum response times for each episode
    avg_response_times = []  # List to store the average response times for each episode
    max_gimbal_smoothness = []  # List to store the maximum gimbal changes for each episode
    avg_gimbal_smoothness = []  # List to store the average gimbal changes for each episode
    max_throttle_smoothness = []  # List to store the maximum throttle changes for each episode
    avg_throttle_smoothness = []  # List to store the average throttle changes for each episode
    avg_cpu_usages = []  # List to store average CPU usage for each episode
    time_taken_to_land = []  # List to store the time taken to land in seconds for each episode

    for episode in range(num_episodes):
        obs = env.reset()
        start_time = time.time()  # Start timing the episode
        process = psutil.Process()  # Start monitoring the process

        # Run the FSM control loop
        final_obs, total_reward, metrics, done = fsm_control_loop(env, render=False)

        # Analyze the results
        max_deviation = max(metrics['deviations']) if metrics['deviations'] else 0
        avg_deviation = np.mean(metrics['deviations']) if metrics['deviations'] else 0
        max_deviations.append(max_deviation)
        avg_deviations.append(avg_deviation)

        max_response_time = max(metrics['response_times']) if metrics['response_times'] else 0
        avg_response_time = np.mean(metrics['response_times']) if metrics['response_times'] else 0
        max_response_times.append(max_response_time)
        avg_response_times.append(avg_response_time)

        max_gimbal_smooth = max(metrics['gimbal_smoothness']) if metrics['gimbal_smoothness'] else 0
        avg_gimbal_smooth = np.mean(metrics['gimbal_smoothness']) if metrics['gimbal_smoothness'] else 0
        max_gimbal_smoothness.append(max_gimbal_smooth)
        avg_gimbal_smoothness.append(avg_gimbal_smooth)

        max_throttle_smooth = max(metrics['throttle_smoothness']) if metrics['throttle_smoothness'] else 0
        avg_throttle_smooth = np.mean(metrics['throttle_smoothness']) if metrics['throttle_smoothness'] else 0
        max_throttle_smoothness.append(max_throttle_smooth)
        avg_throttle_smoothness.append(avg_throttle_smooth)

        avg_cpu_usages.append(np.mean([process.cpu_percent(interval=None)]))

        # Record the time taken to land for the episode
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_to_land.append(time_taken)

        episodes.append(episode)

    # Use the plot_results function from the plot_results module
    plot_results(
        episodes=episodes,
        max_deviations=max_deviations,
        avg_deviations=avg_deviations,
        max_response_times=max_response_times,
        avg_response_times=avg_response_times,
        max_gimbal_smoothness=max_gimbal_smoothness,
        avg_gimbal_smoothness=avg_gimbal_smoothness,
        max_throttle_smoothness=max_throttle_smoothness,
        avg_throttle_smoothness=avg_throttle_smoothness,
        avg_cpu_usages=avg_cpu_usages,
        time_taken_to_land=time_taken_to_land,
        model_type='FSM'  # Specify the model type as 'FSM'
    )

# Run the evaluation
evaluate_fsm_stability_response_time_smoothness_and_time(100)  # Evaluate over 100 episodes
