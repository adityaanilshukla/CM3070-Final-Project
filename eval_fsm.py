import time
import numpy as np
from plot_results import plot_results  # Import the plot_results module
from fsm_controller import fsm_control_loop  # Import the FSM control loop
import io
import sys

def evaluate_fsm(num_episodes=100):
    episodes = []
    max_gimbal_smoothness = []  # List to store the maximum gimbal angles for each episode
    avg_gimbal_smoothness = []  # List to store the average gimbal angles for each episode
    min_gimbal_smoothness = []  # List to store the minimum gimbal angles for each episode
    max_deviations = []  # List to store the maximum deviations for each episode
    avg_deviations = []  # List to store the average deviations for each episode
    min_deviations = []  # List to store the minimum deviations for each episode
    max_response_times = []  # List to store the maximum response times for each episode
    avg_response_times = []  # List to store the average response times for each episode
    min_response_times = []  # List to store the minimum response times for each episode
    max_throttle_smoothness = []  # List to store the maximum throttle settings for each episode
    avg_throttle_smoothness = []  # List to store the average throttle settings for each episode
    min_throttle_smoothness = []  # List to store the minimum throttle settings for each episode
    time_taken_to_land = []  # List to store the time taken to land in seconds for each episode
    landing_successes = []  # List to store landing success for each episode
    x_landing_precision = []  # List to store the x-axis landing precision for each episode

    for episode in range(num_episodes):
        obs = env.reset()
        start_time = time.time()  # Start timing the episode
        landed = False  # Flag to track if the rocket has landed successfully
        episode_response_times = []  # Track response times in this episode
        episode_gimbal_smoothness = []
        episode_deviations = []
        episode_throttle_smoothness = []
        start_correction_time = None  # Track the time to correct a deviation

        # Capture the standard output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Run the FSM control loop and gather flight data for every time step
        flight_data, done = fsm_control_loop(env, render=False)

        # Capture the output and check for landing success
        output = sys.stdout.getvalue()
        if "LANDED!!!!!!!!!" in output:
            landed = True

        # Restore standard output
        sys.stdout = old_stdout

        # Process flight data to calculate metrics
        for timestep_data in flight_data:
            state = timestep_data['state']

            # Monitor gimbal angle (in radians)
            gimbal_angle = state['gimbal']
            episode_gimbal_smoothness.append(gimbal_angle)

            # Monitor angle deviation (can be positive or negative, so no abs())
            angle_deviation = state['angle']
            episode_deviations.append(angle_deviation)

            # Monitor throttle settings (absolute value since throttle can't be negative)
            throttle_setting = abs(state['throttle'])
            episode_throttle_smoothness.append(throttle_setting)

            # Detect a significant deviation and start tracking correction time
            if abs(angle_deviation) > 0.1 and start_correction_time is None:
                start_correction_time = time.time()

            # Track response time (time taken to correct a deviation)
            if start_correction_time is not None and abs(angle_deviation) <= 0.02:
                response_time = time.time() - start_correction_time
                episode_response_times.append(response_time)
                start_correction_time = None  # Reset for next deviation

        # Record x-axis landing precision at the end of the flight
        x_position_at_landing = flight_data[-1]['state']['x']  # Assuming the x position is at flight_data[-1]['state']['x']
        x_landing_precision.append(x_position_at_landing)

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

        # Record the max, min, and average throttle settings for the episode
        max_throttle_setting = max(episode_throttle_smoothness) if episode_throttle_smoothness else 0
        min_throttle_setting = min(episode_throttle_smoothness) if episode_throttle_smoothness else 0
        avg_throttle_setting = np.mean(episode_throttle_smoothness) if episode_throttle_smoothness else 0
        max_throttle_smoothness.append(max_throttle_setting)
        min_throttle_smoothness.append(min_throttle_setting)
        avg_throttle_smoothness.append(avg_throttle_setting)

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

        # Record the time taken to land for the episode
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_to_land.append(time_taken)

        # Record landing success
        landing_successes.append(landed)

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
        max_throttle_smoothness=max_throttle_smoothness,
        avg_throttle_smoothness=avg_throttle_smoothness,
        min_throttle_smoothness=min_throttle_smoothness,
        max_response_times=max_response_times,
        avg_response_times=avg_response_times,
        min_response_times=min_response_times,
        time_taken_to_land=time_taken_to_land,
        x_landing_precision=x_landing_precision,  # Include x-axis landing precision
        model_type='FSM',  # Specify the model type as 'FSM'
        landing_successes=landing_successes,
    )

# Run the evaluation
evaluate_fsm(100)  # Evaluate over 100 episodes
