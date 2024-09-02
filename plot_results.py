import matplotlib.pyplot as plt
import os

def plot_results(episodes, max_deviations, avg_deviations, max_response_times, avg_response_times,
                 max_gimbal_smoothness, avg_gimbal_smoothness, max_throttle_smoothness,
                 avg_throttle_smoothness, avg_cpu_usages, time_taken_to_land, model_type):
    """
    Plot the results of the evaluation for either PPO or FSM models.

    Parameters:
    - episodes: List of episode numbers.
    - max_deviations: List of maximum angle deviations per episode.
    - avg_deviations: List of average angle deviations per episode.
    - max_response_times: List of maximum response times for each episode.
    - avg_response_times: List of average response times for each episode.
    - max_gimbal_smoothness: List of maximum gimbal changes per episode.
    - avg_gimbal_smoothness: List of average gimbal changes per episode.
    - max_throttle_smoothness: List of maximum throttle changes per episode.
    - avg_throttle_smoothness: List of average throttle changes per episode.
    - avg_cpu_usages: List of average CPU usage percentages per episode.
    - time_taken_to_land: List of time taken to land per episode.
    - model_type: String specifying the model type ('PPO' or 'FSM').
    """

    output_dir = os.path.join('plots', 'comparison', model_type)
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the labels and titles based on the model type
    title_suffix = f'Over Episodes ({model_type})'
    max_label_prefix = 'Max' if model_type == 'FSM' else 'Max'
    avg_label_prefix = 'Average'

    # Plotting stability over episodes
    plt.figure()
    plt.plot(episodes, max_deviations, label=f'{max_label_prefix} Angle Deviation Used During Episode', color='red')
    plt.plot(episodes, avg_deviations, label=f'{avg_label_prefix} Angle Deviation', color='blue')
    plt.title(f'Rocket Stability {title_suffix}')
    plt.xlabel('Episode Number')
    plt.ylabel('Angle Deviation (radians)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'stability_max_avg_over_episodes.png'))
    plt.show()

    # Plotting response time over episodes
    plt.figure()
    plt.plot(episodes, max_response_times, label=f'{max_label_prefix} Response Time Used During Episode', color='orange')
    plt.plot(episodes, avg_response_times, label=f'{avg_label_prefix} Response Time', color='green')
    plt.title(f'Rocket Response Time {title_suffix}')
    plt.xlabel('Episode Number')
    plt.ylabel('Response Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'response_time_max_avg_over_episodes.png'))
    plt.show()

    # Plotting gimbal smoothness over episodes
    plt.figure()
    plt.plot(episodes, max_gimbal_smoothness, label=f'{max_label_prefix} Gimbal Angle Used During Episode (radians)', color='purple')
    plt.plot(episodes, avg_gimbal_smoothness, label=f'{avg_label_prefix} Change in Gimbal Angle (radians)', color='cyan')
    plt.title(f'Rocket Gimbal Control Smoothness {title_suffix}')
    plt.xlabel('Episode Number')
    plt.ylabel('Change in Gimbal Angle (radians)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'gimbal_smoothness_max_avg_over_episodes.png'))
    plt.show()

    # Plotting throttle smoothness over episodes
    plt.figure()
    plt.plot(episodes, max_throttle_smoothness, label=f'{max_label_prefix} Throttle Setting Used During Episode', color='brown')
    plt.plot(episodes, avg_throttle_smoothness, label=f'{avg_label_prefix} Change in Throttle Setting', color='orange')
    plt.title(f'Rocket Throttle Control Smoothness {title_suffix}')
    plt.xlabel('Episode Number')
    plt.ylabel('Change in Throttle Setting')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'throttle_smoothness_max_avg_over_episodes.png'))
    plt.show()

    # Plotting CPU usage over episodes
    plt.figure()
    plt.plot(episodes, avg_cpu_usages, label='Average CPU Usage (%)', color='magenta')
    plt.title(f'CPU Usage {title_suffix}')
    plt.xlabel('Episode Number')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cpu_usage_over_episodes.png'))
    plt.show()

    # Plotting time taken to land over episodes
    plt.figure()
    plt.plot(episodes, time_taken_to_land, label='Time Taken to Land (seconds)', color='blue')
    plt.title(f'Time Taken to Land {title_suffix}')
    plt.xlabel('Episode Number')
    plt.ylabel('Time Taken (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'time_taken_to_land_over_episodes.png'))
    plt.show()

