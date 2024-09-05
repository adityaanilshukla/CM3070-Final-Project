from ppo_controller import *

# Run the PPO model
flight_data, done = run_ppo_model(env, model, render=True)
