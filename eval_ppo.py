import time
import gym
import os
import numpy as np
from fsm_controller import get_current_state
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from gym.envs.registration import register
import matplotlib.pyplot as plt
import psutil

# Register the RocketLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Adjust the path as needed
    max_episode_steps=2500,
    reward_threshold=0,
)

# Function to create the environment
def make_env():
    env = gym.make('RocketLander-v0')
    return env

# Load the trained model
model_path = os.path.join("model", "ppo2_RocketLander-v0_2024-08-02024338_step_18000000.zip")
model = PPO2.load(model_path)

# Create the environment
env = DummyVecEnv([make_env])

def evaluate_ppo_model(num_episodes=100):
    output_dir = os.path.join('plots', 'comparison', 'PPO')
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

    angle_threshold = 0.1  # Threshold to detect a significant deviation event
    correction_threshold = 0.02  # Threshold to consider the angle corrected

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_deviations = []  # Track angle deviations in this episode
        episode_response_times = []  # Track response times in this episode
        episode_gimbal_smoothness = []  # Track gimbal changes for smoothness
        episode_throttle_smoothness = []  # Track throttle changes for smoothness
        cpu_usages = []  # Track CPU usage for the episode
        start_correction_time = None
        last_gimbal_action = None
        last_throttle_action = None
        start_time = time.time()  # Start timing the episode
        process = psutil.Process()  # Start monitoring the process

        while not done:
            dt = 1.0 / env.envs[0].metadata['video.frames_per_second']
            current_state = get_current_state(obs[0])  # Assuming the state function is compatible

            # Monitor CPU usage
            cpu_usages.append(process.cpu_percent(interval=None))  # CPU usage in percentage

            # Calculate angle deviation from 0 (stability)
            angle_deviation = abs(obs[0][2])  # Assuming obs[0][2] is the angle

            episode_deviations.append(angle_deviation)

            # Detect a significant deviation
            if angle_deviation > angle_threshold and start_correction_time is None:
                start_correction_time = time.time()

            # If a correction is ongoing and the angle is back within the correction threshold
            if start_correction_time is not None and angle_deviation <= correction_threshold:
                response_time = time.time() - start_correction_time
                episode_response_times.append(response_time)
                start_correction_time = None  # Reset for the next deviation

            # Run the PPO model
            action, _states = model.predict(obs)

            # Determine the structure of `action`
            if np.isscalar(action):
                gimbal_action = action
                throttle_action = action  # Assuming same action for both if scalar
            elif len(action.shape) == 1:
                gimbal_action = action[0]  # Assuming first element is gimbal control
                throttle_action = action[1] if action.shape[0] > 1 else 0  # If there's a second control, otherwise 0
            else:
                # Handle other possible structures
                gimbal_action = action[0][0] if action.shape[1] > 0 else 0
                throttle_action = action[0][1] if action.shape[1] > 1 else 0

            obs, reward, done, info = env.step(action)

            # Monitor control inputs for smoothness
            if last_gimbal_action is not None:
                gimbal_change = abs(gimbal_action - last_gimbal_action)
                episode_gimbal_smoothness.append(gimbal_change)

            if last_throttle_action is not None:
                throttle_change = abs(throttle_action - last_throttle_action)
                episode_throttle_smoothness.append(throttle_change)

            last_gimbal_action = gimbal_action
            last_throttle_action = throttle_action

        # Record the max and average deviations for the episode
        max_deviation = max(episode_deviations)
        avg_deviation = np.mean(episode_deviations)
        max_deviations.append(max_deviation)
        avg_deviations.append(avg_deviation)

        # Record the max and average response times for the episode
        if episode_response_times:
            max_response_time = max(episode_response_times)
            avg_response_time = np.mean(episode_response_times)
        else:
            max_response_time = 0
            avg_response_time = 0

        max_response_times.append(max_response_time)
        avg_response_times.append(avg_response_time)

        # Record the max and average gimbal smoothness for the episode
        if episode_gimbal_smoothness:
            max_gimbal_smooth = max(episode_gimbal_smoothness)
            avg_gimbal_smooth = np.mean(episode_gimbal_smoothness)
        else:
            max_gimbal_smooth = 0
            avg_gimbal_smooth = 0

        max_gimbal_smoothness.append(max_gimbal_smooth)
        avg_gimbal_smoothness.append(avg_gimbal_smooth)

        # Record the max and average throttle smoothness for the episode
        if episode_throttle_smoothness:
            max_throttle_smooth = max(episode_throttle_smoothness)
            avg_throttle_smooth = np.mean(episode_throttle_smoothness)
        else:
            max_throttle_smooth = 0
            avg_throttle_smooth = 0

        max_throttle_smoothness.append(max_throttle_smooth)
        avg_throttle_smoothness.append(avg_throttle_smooth)

        # Record the average CPU usage for the episode
        avg_cpu_usages.append(np.mean(cpu_usages))

        # Record the time taken to land for the episode
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_to_land.append(time_taken)

        episodes.append(episode)

    # Plotting stability over episodes
    plt.figure()
    plt.plot(episodes, max_deviations, label='Max Angle Deviation', color='red')
    plt.plot(episodes, avg_deviations, label='Average Angle Deviation', color='blue')
    plt.title('Rocket Stability Over Episodes (PPO)')
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
    plt.title('Rocket Response Time Over Episodes (PPO)')
    plt.xlabel('Episode Number')
    plt.ylabel('Response Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'response_time_max_avg_over_episodes.png'))
    plt.show()

    # Plotting gimbal smoothness over episodes
    plt.figure()
    plt.plot(episodes, max_gimbal_smoothness, label='Max Change in Gimbal Angle (Degrees)', color='purple')
    plt.plot(episodes, avg_gimbal_smoothness, label='Average Change in Gimbal Angle (Degrees)', color='cyan')
    plt.title('Rocket Gimbal Control Smoothness Over Episodes (PPO)')
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
    plt.title('Rocket Throttle Control Smoothness Over Episodes (PPO)')
    plt.xlabel('Episode Number')
    plt.ylabel('Change in Throttle Setting')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'throttle_smoothness_max_avg_over_episodes.png'))
    plt.show()

    # Plotting CPU usage over episodes
    plt.figure()
    plt.plot(episodes, avg_cpu_usages, label='Average CPU Usage (%)', color='magenta')
    plt.title('CPU Usage Over Episodes (PPO)')
    plt.xlabel('Episode Number')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cpu_usage_over_episodes.png'))
    plt.show()

    # Plotting time taken to land over episodes
    plt.figure()
    plt.plot(episodes, time_taken_to_land, label='Time Taken to Land (seconds)', color='blue')
    plt.title('Time Taken to Land Over Episodes (PPO)')
    plt.xlabel('Episode Number')
    plt.ylabel('Time Taken (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'time_taken_to_land_over_episodes.png'))
    plt.show()

# Run the evaluation
evaluate_ppo_model(100)  # Evaluate over 100 episodes
