import time
import gym
import os
import numpy as np
import psutil  # For monitoring system resources
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from gym.envs.registration import register
from plot_results import plot_results  # Import the plot_results module
from fsm_controller import get_current_state  # Import the state extraction function
import io
import sys

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
    episodes = []
    max_deviations = []  # List to store the maximum deviations for each episode
    avg_deviations = []  # List to store the average deviations for each episode
    min_deviations = []  # List to store the minimum deviations for each episode
    max_response_times = []  # List to store the maximum response times for each episode
    avg_response_times = []  # List to store the average response times for each episode
    min_response_times = []  # List to store the minimum response times for each episode
    max_gimbal_angles = []  # List to store the maximum gimbal angles for each episode
    avg_gimbal_angles = []  # List to store the average gimbal angles for each episode
    min_gimbal_angles = []  # List to store the minimum gimbal angles for each episode
    max_throttle_settings = []  # List to store the maximum throttle settings for each episode
    avg_throttle_settings = []  # List to store the average throttle settings for each episode
    min_throttle_settings = []  # List to store the minimum throttle settings for each episode
    avg_cpu_usages = []  # List to store average CPU usage for each episode
    time_taken_to_land = []  # List to store the time taken to land in seconds for each episode
    landing_successes = []  # List to store landing success for each episode

    angle_threshold = 0.1  # Threshold to detect a significant deviation event
    correction_threshold = 0.02  # Threshold to consider the angle corrected

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_deviations = []  # Track angle deviations in this episode
        episode_response_times = []  # Track response times in this episode
        episode_gimbal_angles = []  # Track gimbal angles in this episode
        episode_throttle_settings = []  # Track throttle settings in this episode
        cpu_usages = []  # Track CPU usage for the episode
        start_correction_time = None
        start_time = time.time()  # Start timing the episode
        process = psutil.Process()  # Start monitoring the process
        landed = False  # Flag to track if the rocket has landed successfully

        # Capture the standard output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        while not done:
            dt = 1.0 / env.envs[0].metadata['video.frames_per_second']
            current_state = get_current_state(obs[0])  # Extract state variables

            # Monitor CPU usage
            cpu_usages.append(process.cpu_percent(interval=None))  # CPU usage in percentage

            # Calculate angle deviation from 0 (stability)
            angle_deviation = abs(current_state['angle'])  # Use angle from the state
            episode_deviations.append(angle_deviation)

            # Monitor gimbal angle (in radians)
            gimbal_angle = abs(current_state['gimbal'])  # Use gimbal from the state
            episode_gimbal_angles.append(gimbal_angle)

            # Monitor throttle settings
            throttle_setting = current_state['throttle']  # Use throttle from the state
            episode_throttle_settings.append(throttle_setting)

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

            obs, reward, done, info = env.step(action)

            # Capture the output
            output = sys.stdout.getvalue()

            # Check for landing success in the captured output
            if "LANDED!!!!!!!!!" in output:
                landed = True

        # Restore the standard output
        sys.stdout = old_stdout

        # Record landing success
        landing_successes.append(landed)

        # Record the max, min, and average deviations for the episode
        max_deviation = max(episode_deviations)
        avg_deviation = np.mean(episode_deviations)
        min_deviation = min(episode_deviations)
        max_deviations.append(max_deviation)
        avg_deviations.append(avg_deviation)
        min_deviations.append(min_deviation)

        # Record the max, min, and average response times for the episode
        if episode_response_times:
            max_response_time = max(episode_response_times)
            avg_response_time = np.mean(episode_response_times)
            min_response_time = min(episode_response_times)
        else:
            max_response_time = 0
            avg_response_time = 0
            min_response_time = 0

        max_response_times.append(max_response_time)
        avg_response_times.append(avg_response_time)
        min_response_times.append(min_response_time)

        # Record the max, min, and average gimbal angles for the episode
        if episode_gimbal_angles:
            max_gimbal_angle = max(episode_gimbal_angles)
            avg_gimbal_angle = np.mean(episode_gimbal_angles)
            min_gimbal_angle = min(episode_gimbal_angles)
        else:
            max_gimbal_angle = 0
            avg_gimbal_angle = 0
            min_gimbal_angle = 0

        max_gimbal_angles.append(max_gimbal_angle)
        avg_gimbal_angles.append(avg_gimbal_angle)
        min_gimbal_angles.append(min_gimbal_angle)

        # Record the max, min, and average throttle settings for the episode
        if episode_throttle_settings:
            max_throttle_setting = max(episode_throttle_settings)
            avg_throttle_setting = np.mean(episode_throttle_settings)
            min_throttle_setting = min(episode_throttle_settings)
        else:
            max_throttle_setting = 0
            avg_throttle_setting = 0
            min_throttle_setting = 0

        max_throttle_settings.append(max_throttle_setting)
        avg_throttle_settings.append(avg_throttle_setting)
        min_throttle_settings.append(min_throttle_setting)

        # Record the average CPU usage for the episode
        avg_cpu_usages.append(np.mean(cpu_usages))

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
        min_deviations=min_deviations,
        max_response_times=max_response_times,
        avg_response_times=avg_response_times,
        min_response_times=min_response_times,
        max_gimbal_smoothness=max_gimbal_angles,  # Plot gimbal angles directly
        avg_gimbal_smoothness=avg_gimbal_angles,  # Plot average gimbal angles
        min_gimbal_smoothness=min_gimbal_angles,  # Plot minimum gimbal angles
        max_throttle_smoothness=max_throttle_settings,  # Separate throttle settings
        avg_throttle_smoothness=avg_throttle_settings,  # Separate average throttle settings
        min_throttle_smoothness=min_throttle_settings,  # Separate minimum throttle settings
        avg_cpu_usages=avg_cpu_usages,
        time_taken_to_land=time_taken_to_land,
        model_type='PPO',  # Specify the model type as 'PPO'
        landing_successes=landing_successes  # Include landing successes in the plot
    )

# Run the evaluation
evaluate_ppo_model(100)  # Evaluate over 100 episodes

