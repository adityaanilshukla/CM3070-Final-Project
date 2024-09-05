import gym
import os
from stable_baselines import PPO2
from gym.envs.registration import register
from fsm_controller import get_current_state  # Import the state extraction function

# Register the RocketLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Adjust the path as needed
    max_episode_steps=3000,
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
env = make_env()

def run_ppo_model(env, model, render=False):
    """
    Run the PPO model in the RocketLander environment and return flight data.

    Parameters:
    - env: The RocketLander environment
    - model: The trained PPO model
    - render: Boolean flag to indicate whether to render the environment

    Returns:
    - flight_data: A list of dictionaries containing 'state' and 'obs' at each step
    - done: Boolean flag indicating if the episode is finished
    """
    obs = env.reset()
    done = False
    step = 0
    flight_data = []  # To store the state and observation at each step

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Extract current state variables using get_current_state
        try:
            current_state = get_current_state(obs)  # No [0] index
        except IndexError:
            current_state = {}  # Handle case where state cannot be extracted

        # Append the state and obs to flight_data
        if current_state:
            flight_data.append({
                'state': current_state,   # The extracted state variables
                'obs': obs                # The raw observation array
            })

        # Render the environment based on the render flag
        if render:
            env.render()

        step += 1

    env.close()

    return flight_data, done
