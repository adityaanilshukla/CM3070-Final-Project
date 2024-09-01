import time
import numpy as np
import gym
import psutil  # For monitoring system resources
from fsm_controller import *
from gym.envs.registration import register
import matplotlib.pyplot as plt
import os

env = gym.make('RocketLander-v0')

def evaluate_fsm_stability_response_time_smoothness_and_time(num_episodes=100):
    output_dir = os.path.join('plots', 'comparison', 'FSM')
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
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

    # Plotting stability over episodes
    plt.figure()
    plt.plot(episodes, max_deviations, label='Max Angle Deviation', color='red')
    plt.plot(episodes, avg_deviations, label='Average Angle Deviation', color='blue')
    plt.title('Rocket Stability Over Episodes (FSM)')
    plt.xlabel('Episode Number')
    plt.ylabel('Angle Deviation (radians)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'stability_max_avg_over_episodes.png'))
    plt.show()

    # Plotting response time over episodes
    plt.figure()
    plt.plot(episodes, max_response_times, label='Max Response Time', color='orange')
    plt.plot(episodes, avg_response_times, label='Average Response Time', color='green')
    plt.title('Rocket Response Time Over Episodes (FSM)')
    plt.xlabel('Episode Number')
    plt.ylabel('Response Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'response_time_max_avg_over_episodes.png'))
    plt.show()

    # Plotting gimbal smoothness over episodes
    plt.figure()
    plt.plot(episodes, max_gimbal_smoothness, label='Max Gimbal Angle (Degrees)', color='purple')
    plt.plot(episodes, avg_gimbal_smoothness, label='Average Change in Gimbal Angle (Degrees)', color='cyan')
    plt.title('Rocket Gimbal Control Smoothness Over Episodes (FSM)')
    plt.xlabel('Episode Number')
    plt.ylabel('Change in Gimbal Angle (Degrees)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'gimbal_smoothness_max_avg_over_episodes.png'))
    plt.show()

    # Plotting throttle smoothness over episodes
    plt.figure()
    plt.plot(episodes, max_throttle_smoothness, label='Max Throttle Setting', color='brown')
    plt.plot(episodes, avg_throttle_smoothness, label='Average Change in Throttle Setting', color='orange')
    plt.title('Rocket Throttle Control Smoothness Over Episodes (FSM)')
    plt.xlabel('Episode Number')
    plt.ylabel('Change in Throttle Setting')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'throttle_smoothness_max_avg_over_episodes.png'))
    plt.show()

    # Plotting CPU usage over episodes
    plt.figure()
    plt.plot(episodes, avg_cpu_usages, label='Average CPU Usage (%)', color='magenta')
    plt.title('CPU Usage Over Episodes (FSM)')
    plt.xlabel('Episode Number')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cpu_usage_over_episodes.png'))
    plt.show()

    # Plotting time taken to land over episodes
    plt.figure()
    plt.plot(episodes, time_taken_to_land, label='Time Taken to Land (seconds)', color='blue')
    plt.title('Time Taken to Land Over Episodes (FSM)')
    plt.xlabel('Episode Number')
    plt.ylabel('Time Taken (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'time_taken_to_land_over_episodes.png'))
    plt.show()

# Run the evaluation
evaluate_fsm_stability_response_time_smoothness_and_time(100)  # Evaluate over 100 episodes
