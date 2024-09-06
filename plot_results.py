import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(episodes, max_deviations, avg_deviations, min_deviations,
                 max_response_times, avg_response_times, min_response_times,
                 max_gimbal_smoothness, avg_gimbal_smoothness, min_gimbal_smoothness,
                 max_throttle_smoothness, avg_throttle_smoothness, min_throttle_smoothness,
                 time_taken_to_land, model_type, landing_successes, x_landing_precision,
                 ram_usage, action_counts):
    """
    Plot the results of the evaluation for either PPO or FSM models, including a pie chart
    for the action distribution.

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
    - ram_usage: List of RAM usage per episode.
    - action_counts: Dictionary containing the count of each action (0 to 6).
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
        plt.figure(figsize=(16, 8))  # Increase figure size for better readability
        plt.plot(episodes, max_deviations, label=f'{max_label_prefix} Angle Deviation Used During Episode', color='red')
        plt.plot(episodes, avg_deviations, label=f'{avg_label_prefix} Angle Deviation', color='blue')
        plt.plot(episodes, min_deviations, label=f'{min_label_prefix} Angle Deviation Used During Episode', color='green')
        plt.title(f'Rocket Stability {title_suffix}', fontsize=16)
        plt.xlabel('Episode Number', fontsize=14)
        plt.ylabel('Angle Deviation (radians)', fontsize=14)
        plt.grid(True)
        # Adjust the font size of the legend and place it outside the plot area
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
        # Adjust plot size to accommodate the legend and stats box
        plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
        # Adding statistics box and placing it outside the plot, to the right
        stats_text = (f'Mean: {np.mean(avg_deviations):.2f}\n'
                       f'Standard Deviation: {np.std(avg_deviations):.2f}\n'
                       f'Variance: {np.var(avg_deviations):.2f}')
        plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Save the image with a unique name
        plt.savefig(os.path.join(output_dir, 'stability_over_episodes.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Plotting response time over episodes
    if max_response_times:
        plt.figure(figsize=(16, 8))  # Increase figure size for better readability
        plt.plot(episodes, max_response_times, label=f'{max_label_prefix} Response Time Used During Episode', color='orange')
        plt.plot(episodes, avg_response_times, label=f'{avg_label_prefix} Response Time', color='green')
        plt.plot(episodes, min_response_times, label=f'{min_label_prefix} Response Time Used During Episode', color='blue')
        plt.title(f'Rocket Response Time {title_suffix}', fontsize=16)
        plt.xlabel('Episode Number')
        plt.ylabel('Response Time (seconds)')
        plt.grid(True)
        # Adjust the font size of the legend and place it outside the plot area
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
        # Adjust plot size to accommodate the legend and stats box
        plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
        # Adding statistics box and placing it outside the plot, to the right
        stats_text = (f'Mean: {np.mean(avg_response_times):.2f}\n'
                       f'Standard Deviation: {np.std(avg_response_times):.2f}\n'
                       f'Variance: {np.var(avg_response_times):.2f}')
        plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Save the image with a unique name
        plt.savefig(os.path.join(output_dir, 'response_time_over_episodes.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Plotting gimbal smoothness over episodes
    if max_gimbal_smoothness:
        plt.figure(figsize=(16, 8))  # Increase figure size for better readability
        plt.plot(episodes, max_gimbal_smoothness, label=f'{max_label_prefix} Gimbal Angle Used During Episode (radians)', color='purple')
        plt.plot(episodes, avg_gimbal_smoothness, label=f'{avg_label_prefix} Change in Gimbal Angle (radians)', color='cyan')
        plt.plot(episodes, min_gimbal_smoothness, label=f'{min_label_prefix} Gimbal Angle Used During Episode (radians)', color='green')
        plt.title(f'Rocket Gimbal Control Smoothness {title_suffix}', fontsize=16)
        plt.xlabel('Episode Number')
        plt.ylabel('Change in Gimbal Angle (radians)')
        plt.grid(True)
        # Adjust the font size of the legend and place it outside the plot area
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
        # Adjust plot size to accommodate the legend and stats box
        plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)

        # Adding statistics box and placing it outside the plot, to the right
        stats_text = (f'Mean: {np.mean(avg_gimbal_smoothness):.2f}\n'
                       f'Standard Deviation: {np.std(avg_gimbal_smoothness):.2f}\n'
                       f'Variance: {np.var(avg_gimbal_smoothness):.2f}')
        plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Save the image with a unique name
        plt.savefig(os.path.join(output_dir, 'gimbal_smoothness_over_episodes.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Plotting throttle smoothness over episodes
    if max_throttle_smoothness:
        plt.figure(figsize=(16, 8))  # Increase figure size for better readability
        plt.plot(episodes, max_throttle_smoothness, label=f'{max_label_prefix} Throttle Setting Used During Episode', color='brown')
        plt.plot(episodes, avg_throttle_smoothness, label=f'{avg_label_prefix} Change in Throttle Setting', color='orange')
        plt.plot(episodes, min_throttle_smoothness, label=f'{min_label_prefix} Throttle Setting Used During Episode', color='green')
        plt.title(f'Rocket Throttle Control Smoothness {title_suffix}', fontsize=16)
        plt.xlabel('Episode Number')
        plt.ylabel('Change in Throttle Setting')
        plt.grid(True)
        plt.legend()
        # Adjust plot size to accommodate the legend and stats box
        plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
        # Adding statistics box
        stats_text = (f'Mean: {np.mean(avg_throttle_smoothness):.2f}\n'
                      f'Standard Deviation: {np.std(avg_throttle_smoothness):.2f}\n'
                      f'Variance: {np.var(avg_throttle_smoothness):.2f}')
        plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Save the image with a unique name
        plt.savefig(os.path.join(output_dir, 'throttle_smoothness_over_episodes.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Plotting time taken to land over episodes
    if time_taken_to_land:
        plt.figure(figsize=(16, 8))
        plt.plot(episodes, time_taken_to_land, label='Time Taken to Land (seconds)', color='blue')
        plt.title(f'Time Taken to Land {title_suffix}', fontsize=16)
        plt.xlabel('Episode Number', fontsize=14)
        plt.ylabel('Time Taken (seconds)', fontsize=14)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
        plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
        # Adding statistics box and placing it outside the plot, to the right
        stats_text = (f'Mean: {np.mean(time_taken_to_land):.2f}\n'
                      f'Standard Deviation: {np.std(time_taken_to_land):.2f}\n'
                      f'Variance: {np.var(time_taken_to_land):.2f}')
        plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Save the image with a unique name
        plt.savefig(os.path.join(output_dir, 'time_taken_to_land_over_episodes.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Plotting x-axis landing precision over episodes
    if x_landing_precision:
        plt.figure(figsize=(16, 8))
        plt.plot(episodes, x_landing_precision, label='X Landing Precision', color='magenta')
        plt.title(f'X Landing Precision {title_suffix}', fontsize=16)
        plt.xlabel('Episode Number', fontsize=14)
        plt.ylabel('X Position at Landing (closeness to 0)', fontsize=14)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
        plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
        # Adding statistics box and placing it outside the plot, to the right
        stats_text = (f'Mean: {np.mean(x_landing_precision):.2f}\n'
                      f'Standard Deviation: {np.std(x_landing_precision):.2f}\n'
                      f'Variance: {np.var(x_landing_precision):.2f}')
        plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Save the image with a unique name
        plt.savefig(os.path.join(output_dir, 'x_landing_precision_over_episodes.png'), bbox_inches='tight', dpi=300)
        plt.close

    # Plotting landing successes and failures
    if landing_successes:
        success_count = sum(landing_successes)
        failure_count = len(landing_successes) - success_count

        plt.figure(figsize=(16, 8))
        plt.bar(['Successes', 'Failures'], [success_count, failure_count], color=['green', 'red'])
        plt.title(f'Landing Successes vs Failures {title_suffix}', fontsize=16)
        plt.ylabel('Number of Episodes', fontsize=14)
        plt.text(0, success_count + 0.5, f'Successes: {success_count}', ha='center', va='bottom', fontsize=12)
        plt.text(1, failure_count + 0.5, f'Failures: {failure_count}', ha='center', va='bottom', fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'landing_successes_vs_failures.png'), bbox_inches='tight', dpi=300)
        plt.close()

    #plotting RAM usage over episodes
    if ram_usage:
        plt.figure(figsize=(16, 8))
        plt.plot(episodes, ram_usage, label='RAM Usage (MB)', color='magenta')
        plt.title(f'RAM Usage {title_suffix}', fontsize=16)
        plt.xlabel('Episode Number', fontsize=14)
        plt.ylabel('RAM Usage (MB)', fontsize=14)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
        plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
        # Adding statistics box and placing it outside the plot, to the right
        stats_text = (f'Mean: {np.mean(ram_usage):.2f}\n'
                      f'Standard Deviation: {np.std(ram_usage):.2f}\n'
                      f'Variance: {np.var(ram_usage):.2f}')
        plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Save the image with a unique name
        plt.savefig(os.path.join(output_dir, 'ram_usage_over_episodes.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # if action_counts:
    #     # Create a pie chart for action distribution (excluding action 6 - "no action")
    #     total_actions = sum(action_counts[action] for action in range(6))  # Exclude action 6
    #     if total_actions > 0:
    #         action_labels = ['Gimbal Left', 'Gimbal Right', 'Throttle Up', 'Throttle Down',
    #                          'First Control Thruster', 'Second Control Thruster']
    #         action_values = [action_counts[action] for action in range(6)]
    #
    #         # Calculate the percentage breakdown of actions
    #         action_percentages = [100 * value / total_actions for value in action_values]
    #
    #         # Define colors for each action
    #         colors = plt.cm.Paired.colors[:6]  # Using a color palette with six colors
    #         
    #         # Create the pie chart without the labels
    #         plt.figure()
    #         wedges, _ = plt.pie(action_percentages, colors=colors, startangle=90)
    #
    #         # Create the legend and stats box
    #         legend_labels = [f'{label}: {percent:.1f}%' for label, percent in zip(action_labels, action_percentages)]
    #         
    #         # Positioning the legend box
    #         plt.legend(wedges, legend_labels, title="Actions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    #         
    #         # Calculate the mean number of actions per episode (for actions 0 to 5)
    #         mean_actions_per_episode = total_actions / len(episodes)
    #         
    #         # Adding statistics box
    #         stats_text = (f'Mean number of actions per episode (excluding no action): {mean_actions_per_episode:.2f}')
    #         plt.gcf().text(0.74, 0.2, stats_text, bbox=dict(facecolor='white', alpha=0.5))
    #
    #         # Save the pie chart
    #         plt.title(f'Action Distribution {title_suffix}')
    #         plt.savefig(os.path.join(output_dir, 'action_distribution_pie_chart.png'))
    #         plt.close()
    # 

    if action_counts:
        # Create a pie chart for action distribution (excluding action 6 - "no action")
        total_actions = sum(action_counts[action] for action in range(6))  # Exclude action 6
        if total_actions > 0:
            action_labels = ['Gimbal Left', 'Gimbal Right', 'Throttle Up', 'Throttle Down',
                             'First Control Thruster', 'Second Control Thruster']
            action_values = [action_counts[action] for action in range(6)]

            # Calculate the percentage breakdown of actions
            action_percentages = [100 * value / total_actions for value in action_values]

            # Define colors for each action
            colors = plt.cm.Paired.colors[:6]  # Using a color palette with six colors

            # Create the pie chart without the labels
            plt.figure(figsize=(16, 8))  # Match the figure size for consistency
            wedges, _ = plt.pie(action_percentages, colors=colors, startangle=90)

            # Positioning the legend box outside the plot, on the right
            legend_labels = [f'{label}: {percent:.1f}%' for label, percent in zip(action_labels, action_percentages)]
            plt.legend(wedges, legend_labels, title="Actions", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)

            # Calculate the mean number of actions per episode (for actions 0 to 5)
            mean_actions_per_episode = total_actions / len(episodes)

            # Adding statistics box and placing it outside the plot, to the right
            stats_text = (f'Mean number of actions per episode (excluding no action): {mean_actions_per_episode:.2f}')
            plt.gcf().text(0.77, 0.3, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            # Adjust plot size to accommodate the legend and stats box
            plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)

            # Save the pie chart
            plt.title(f'Action Distribution {title_suffix}', fontsize=16)
            plt.savefig(os.path.join(output_dir, 'action_distribution_pie_chart.png'), bbox_inches='tight', dpi=300)
            plt.close()
