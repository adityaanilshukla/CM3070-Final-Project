import time
import numpy as np
import psutil  # For monitoring system resources
import gc  # Garbage collection
from plot_results import plot_results  # Import the plot_results module
from fsm_controller import *  # Import the FSM control loop
import io
import sys

def evaluate_fsm(num_episodes=100):
    episodes = []
    max_gimbal_smoothness = [] 
    avg_gimbal_smoothness = []
    min_gimbal_smoothness = []
    max_deviations = []
    avg_deviations = []
    min_deviations = []
    max_response_times = []
    avg_response_times = []
    min_response_times = []
    max_throttle_smoothness = []
    avg_throttle_smoothness = []
    min_throttle_smoothness = []
    time_taken_to_land = []
    landing_successes = []
    x_landing_precision = []
    ram_usage = []

    for episode in range(num_episodes):
        # Reinitialize the environment for every episode
        global env
        env.close()  # Close the existing environment
        del env  # Delete it to free memory
        env = gym.make('RocketLander-v0')  # Recreate the environment

        obs = env.reset()
        start_time = time.time()
        process = psutil.Process()
        landed = False
        episode_response_times = []
        episode_gimbal_smoothness = []
        episode_deviations = []
        episode_throttle_smoothness = []
        start_correction_time = None

        # Capture the standard output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Run the FSM control loop and gather flight data for every time step
        flight_data, done = fsm_control_loop(env, render=False)

        output = sys.stdout.getvalue()
        if "LANDED!!!!!!!!!" in output:
            landed = True

        sys.stdout = old_stdout

        ram_usage.append(process.memory_info().rss / (1024 ** 2))

        for timestep_data in flight_data:
            state = timestep_data['state']

            gimbal_angle = state['gimbal']
            episode_gimbal_smoothness.append(gimbal_angle)

            angle_deviation = state['angle']
            episode_deviations.append(angle_deviation)

            throttle_setting = abs(state['throttle'])
            episode_throttle_smoothness.append(throttle_setting)

            if abs(angle_deviation) > 0.1 and start_correction_time is None:
                start_correction_time = time.time()

            if start_correction_time is not None and abs(angle_deviation) <= 0.02:
                response_time = time.time() - start_correction_time
                episode_response_times.append(response_time)
                start_correction_time = None

        x_position_at_landing = flight_data[-1]['state']['x']
        x_landing_precision.append(x_position_at_landing)

        max_gimbal_angle = max(episode_gimbal_smoothness) if episode_gimbal_smoothness else 0
        min_gimbal_angle = min(episode_gimbal_smoothness) if episode_gimbal_smoothness else 0
        avg_gimbal_angle = np.mean(episode_gimbal_smoothness) if episode_gimbal_smoothness else 0
        max_gimbal_smoothness.append(max_gimbal_angle)
        min_gimbal_smoothness.append(min_gimbal_angle)
        avg_gimbal_smoothness.append(avg_gimbal_angle)

        max_deviation = max(episode_deviations) if episode_deviations else 0
        min_deviation = min(episode_deviations) if episode_deviations else 0
        avg_deviation = np.mean(episode_deviations) if episode_deviations else 0
        max_deviations.append(max_deviation)
        min_deviations.append(min_deviation)
        avg_deviations.append(avg_deviation)

        max_throttle_setting = max(episode_throttle_smoothness) if episode_throttle_smoothness else 0
        min_throttle_setting = min(episode_throttle_smoothness) if episode_throttle_smoothness else 0
        avg_throttle_setting = np.mean(episode_throttle_smoothness) if episode_throttle_smoothness else 0
        max_throttle_smoothness.append(max_throttle_setting)
        min_throttle_smoothness.append(min_throttle_setting)
        avg_throttle_smoothness.append(avg_throttle_setting)

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

        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_to_land.append(time_taken)

        landing_successes.append(landed)

        episodes.append(episode)

        # Cleanup memory and force garbage collection
        del flight_data, episode_gimbal_smoothness, episode_deviations, episode_throttle_smoothness, episode_response_times
        gc.collect()

    plot_results(
        episodes=episodes,
        max_gimbal_smoothness=max_gimbal_smoothness,
        min_gimbal_smoothness=min_gimbal_smoothness,
        avg_gimbal_smoothness=avg_gimbal_smoothness,
        max_deviations=max_deviations,
        avg_deviations=avg_deviations,
        min_deviations=min_deviations,
        max_throttle_smoothness=max_throttle_smoothness,
        avg_throttle_smoothness=avg_throttle_smoothness,
        min_throttle_smoothness=min_throttle_smoothness,
        max_response_times=max_response_times,
        avg_response_times=avg_response_times,
        min_response_times=min_response_times,
        time_taken_to_land=time_taken_to_land,
        x_landing_precision=x_landing_precision,
        model_type='FSM',
        landing_successes=landing_successes,
        ram_usage=ram_usage  # Include RAM usage in the plot
    )

# Run the evaluation
evaluate_fsm(100)
