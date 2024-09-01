from fsm_controller import * 

# run fsm model for one iteration
final_obs, total_reward, metrics, done = fsm_control_loop(env, render=True)
