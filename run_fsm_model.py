from fsm_controller import * 

# run fsm model for one iteration
flight_data, done = fsm_control_loop(env, render=True)