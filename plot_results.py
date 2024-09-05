import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(episodes, max_deviations, avg_deviations, min_deviations,
                 max_response_times, avg_response_times, min_response_times,
                 max_gimbal_smoothness, avg_gimbal_smoothness, min_gimbal_smoothness,
                 max_throttle_smoothness, avg_throttle_smoothness, min_throttle_smoothness,
                 time_taken_to_land, model_type, landing_successes, x_landing_precision):
    """
    Plot the results of the evaluation for either PPO or FSM models.

    Parameters:
    - episodes: List of episode numbers.
    - max_deviations: List of maximum angle deviations per episode.
    - avg_deviations: List of average angle deviations per episode.
    - min_deviations: List of minimum angle deviations per episode.
    - max_response_times: List of maximum response times per episode.
    - avg_response_times: List of average response times per episode.
    - min_response_times: List of minimum response times per episode.
    - max_gimbal_smoothness: List of maximum gimbal changes per episode.
    - avg_gimbal_smoothness: List of average gimbal changes per episode.
    - min_gimbal_smoothness: List of minimum gimbal changes per episode.
    - max_throttle_smoothness: List of maximum throttle changes per episode.
    - avg_throttle_smoothness: List of average throttle changes per episode.
    - min_throttle_smoothness: List of minimum throttle changes per episode.
    - time_taken_to_land: List of time taken to land per episode.
    - model_type: String specifying the model type ('PPO' or 'FSM').
    - landing_successes: List of boolean values indicating success (True) or failure (False) for each episode.
    - x_landing_precision: List of x-axis positions at the end of each episode.
    """

    output_dir = os.path.join('plots', 'comparison', model_type)
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the labels and titles based on the model type
    title_suffix = f'Over Episodes ({model_type})'
    max_label_prefix = 'Max' if model_type == 'FSM' else 'Max'
    avg_label_prefix = 'Average'
    min_label_prefix = 'Min'

    # Plotting stability over episodes
    if max_deviations:
        plt.figure()
        plt.plot(episodes, max_deviations, label=f'{max_label_prefix} Angle Deviation Used During Episode', color='red')
        plt.plot(episodes, avg_deviations, label=f'{avg_label_prefix} Angle Deviation', color='blue')
        plt.plot(episodes, min_deviations, label=f'{min_label_prefix} Angle Deviation Used During Episode', color='green')
        plt.title(f'Rocket Stability {title_suffix}')
        plt.xlabel('Episode Number')
        plt.ylabel('Angle Deviation (radians)')
        plt.grid(True)
        plt.legend()
        plt.subplots_adjust(right=0.74)  # Make room for stats box

        # Adding statistics box
        stats_text = (f'Mean: {np.mean(avg_deviations):.2f}\n'
                      f'Standard Deviation: {np.std(avg_deviations):.2f}\n'
                      f'Variance: {np.var(avg_deviations):.2f}')
        plt.gcf().text(0.74, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.5))  # Adjust position
        plt.savefig(os.path.join(output_dir, 'stability_max_avg_min_over_episodes.png'))
        plt.close()

    # Plotting response time over episodes
    if max_response_times:
        plt.figure()
        plt.plot(episodes, max_response_times, label=f'{max_label_prefix} Response Time Used During Episode', color='orange')
        plt.plot(episodes, avg_response_times, label=f'{avg_label_prefix} Response Time', color='green')
        plt.plot(episodes, min_response_times, label=f'{min_label_prefix} Response Time Used During Episode', color='blue')
        plt.title(f'Rocket Response Time {title_suffix}')
        plt.xlabel('Episode Number')
        plt.ylabel('Response Time (seconds)')
        plt.grid(True)
        plt.legend()
        plt.subplots_adjust(right=0.74)  # Make room for stats box

        # Adding statistics box
        stats_text = (f'Mean: {np.mean(avg_response_times):.2f}\n'
                      f'Standard Deviation: {np.std(avg_response_times):.2f}\n'
                      f'Variance: {np.var(avg_response_times):.2f}')
        plt.gcf().text(0.74, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.5))  # Adjust position
        plt.savefig(os.path.join(output_dir, 'response_time_max_avg_min_over_episodes.png'))
        plt.close()

    # Plotting gimbal smoothness over episodes
    if max_gimbal_smoothness:
        plt.figure()
        plt.plot(episodes, max_gimbal_smoothness, label=f'{max_label_prefix} Gimbal Angle Used During Episode (radians)', color='purple')
        plt.plot(episodes, avg_gimbal_smoothness, label=f'{avg_label_prefix} Change in Gimbal Angle (radians)', color='cyan')
        plt.plot(episodes, min_gimbal_smoothness, label=f'{min_label_prefix} Gimbal Angle Used During Episode (radians)', color='green')
        plt.title(f'Rocket Gimbal Control Smoothness {title_suffix}')
        plt.xlabel('Episode Number')
        plt.ylabel('Change in Gimbal Angle (radians)')
        plt.grid(True)
        plt.legend()
        plt.subplots_adjust(right=0.74)  # Make room for stats box

        # Adding statistics box
        stats_text = (f'Mean: {np.mean(avg_gimbal_smoothness):.2f}\n'
                      f'Standard Deviation: {np.std(avg_gimbal_smoothness):.2f}\n'
                      f'Variance: {np.var(avg_gimbal_smoothness):.2f}')
        plt.gcf().text(0.74, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.5))  # Adjust position
        plt.savefig(os.path.join(output_dir, 'gimbal_smoothness_max_avg_min_over_episodes.png'))
        plt.close()

    # Plotting throttle smoothness over episodes
    if max_throttle_smoothness:
        plt.figure()
        plt.plot(episodes, max_throttle_smoothness, label=f'{max_label_prefix} Throttle Setting Used During Episode', color='brown')
        plt.plot(episodes, avg_throttle_smoothness, label=f'{avg_label_prefix} Change in Throttle Setting', color='orange')
        plt.plot(episodes, min_throttle_smoothness, label=f'{min_label_prefix} Throttle Setting Used During Episode', color='green')
        plt.title(f'Rocket Throttle Control Smoothness {title_suffix}')
        plt.xlabel('Episode Number')
        plt.ylabel('Change in Throttle Setting')
        plt.grid(True)
        plt.legend()
        plt.subplots_adjust(right=0.74)  # Make room for stats box

        # Adding statistics box
        stats_text = (f'Mean: {np.mean(avg_throttle_smoothness):.2f}\n'
                      f'Standard Deviation: {np.std(avg_throttle_smoothness):.2f}\n'
                      f'Variance: {np.var(avg_throttle_smoothness):.2f}')
        plt.gcf().text(0.74, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.5))  # Adjust position
        plt.savefig(os.path.join(output_dir, 'throttle_smoothness_max_avg_min_over_episodes.png'))
        plt.close()

    # Plotting time taken to land over episodes
    if time_taken_to_land:
        plt.figure()
        plt.plot(episodes, time_taken_to_land, label='Time Taken to Land (seconds)', color='blue')
        plt.title(f'Time Taken to Land {title_suffix}')
        plt.xlabel('Episode Number')
        plt.ylabel('Time Taken (seconds)')
        plt.grid(True)
        plt.legend()
        plt.subplots_adjust(right=0.74)  # Make room for stats box

        # Adding statistics box
        stats_text = (f'Mean: {np.mean(time_taken_to_land):.2f}\n'
                      f'Standard Deviation: {np.std(time_taken_to_land):.2f}\n'
                      f'Variance: {np.var(time_taken_to_land):.2f}')
        plt.gcf().text(0.74, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.5))  # Adjust position
        plt.savefig(os.path.join(output_dir, 'time_taken_to_land_over_episodes.png'))
        plt.close()

    # Plotting x-axis landing precision over episodes
    if x_landing_precision:
        plt.figure()
        plt.plot(episodes, x_landing_precision, label='X Landing Precision', color='magenta')
        plt.title(f'X Landing Precision {title_suffix}')
        plt.xlabel('Episode Number')
        plt.ylabel('X Position at Landing (closeness to 0)')
        plt.grid(True)
        plt.legend()
        plt.subplots_adjust(right=0.74)  # Make room for stats box

        # Adding statistics box
        stats_text = (f'Mean: {np.mean(x_landing_precision):.2f}\n'
                      f'Standard Deviation: {np.std(x_landing_precision):.2f}\n'
                      f'Variance: {np.var(x_landing_precision):.2f}')
        plt.gcf().text(0.74, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.5))  # Adjust position
        plt.savefig(os.path.join(output_dir, 'x_landing_precision_over_episodes.png'))
        plt.close()

    # Plotting landing successes and failures
    if landing_successes:
        success_count = sum(landing_successes)
        failure_count = len(landing_successes) - success_count

        plt.figure()
        plt.bar(['Successes', 'Failures'], [success_count, failure_count], color=['green', 'red'])
        plt.title(f'Landing Successes vs Failures {title_suffix}')
        plt.ylabel('Number of Episodes')
        plt.text(0, success_count + 0.5, f'Successes: {success_count}', ha='center', va='bottom')
        plt.text(1, failure_count + 0.5, f'Failures: {failure_count}', ha='center', va='bottom')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'landing_successes_vs_failures.png'))
        plt.close()
