import numpy as np
import gym
import time
from gym.envs.registration import register

# Register the custom environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',
    max_episode_steps=3000,
    reward_threshold=0,
)

env = gym.make('RocketLander-v0')

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.reset()

    def reset(self):
        self.integral = 0
        self.previous_error = None

    def compute(self, error, dt):
        self.integral += error * dt
        if self.previous_error is None:
            derivative = 0
        else:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        lower, upper = self.output_limits
        if lower is not None:
            output = max(lower, output)
        if upper is not None:
            output = min(upper, output)
        return output

# PID controllers for each control variable
pid_angle = PIDController(1200.0, 0.0, 800.0, output_limits=(-3.5, 3.5))
pid_throttle = PIDController(400.0, 0.0, 200.0, output_limits=(-1.27, 0.96))

def set_throttle(obs,dt):
    target_vy = 3 # target descent rate

    x = obs[0]  # x position
    throttle = obs[5]  # Throttle
    vx = obs[7]  # Horizontal velocity
    vy = obs[8]  # Vertical velocity

    # Increase the target descent rate when outside the horizontal velocity threshold
    if abs(vx) >= 0.1 and vy < 0.1:
        target_vy = 12
    # Increase the target descent rate when the rocket is far from the landing zone
    if abs(x) > 0 and vy < 0.1:
        target_vy = 45

    vertical_velocity_error = target_vy - vy
    desired_throttle = pid_throttle.compute(vertical_velocity_error, dt)

    if throttle < desired_throttle:
        return 2 # Throttle up
    elif throttle > desired_throttle:
        return 3 # Throttle down
    else:
        return 6 # Do nothing

def correct_angle(obs, dt):
    x = obs[0]  # x position
    angle = obs[2]  # Angle of the rocket
    gimbal = obs[6]  # Gimbal angle
    vx = obs[7]  # Horizontal velocity

    # Set limits for x position and velocity
    position_tolerance = 0.1  # Acceptable range for x position
    velocity_tolerance = 0.2  # Acceptable range for horizontal velocity
    
    # Increase the aggression factor for angle correction based on distance and velocity
    position_factor = min(max(abs(x) * 0.5, 1), 5)  # Amplify the correction based on distance
    velocity_factor = min(max(abs(vx) * 0.7, 1), 5)  # Amplify the correction based on speed

    # Calculate the target angle with increased aggression when far off
    if abs(x) > position_tolerance or abs(vx) > velocity_tolerance:
        max_angle = np.clip(x * position_factor + vx * velocity_factor, -0.6, 0.6)
    else:
        max_angle = 0.0

    # Calculate the error based on the dynamically determined target angle
    angle_error = max_angle - angle
    desired_gimbal = pid_angle.compute(angle_error, dt)

    # Set the gimbal action based on the desired angle correction
    if desired_gimbal > gimbal:
        return 1 # Gimbal right
    elif desired_gimbal < gimbal:
        return 0 # Gimbal left
    else:
        return 6 # Do nothing

def within_landing_zone(obs,dt):
    x = obs[0]
    y = obs[1]
    angle = obs[2]
    vx = obs[7]
    # Check if the rocket is in the landing zone
    if y <= -1.3 and abs(x) <= 0.3 and abs(vx) <= 0.05:
        return True

def moving_toward_landing_zone(obs,dt):
    # Check if the rocket is moving toward the landing zone
    if abs(obs[7]) <= 0.05 or abs(obs[0]) >= 0:
        return True
    else:
        return False

def land_rocket(obs, dt):
    x = obs[0]
    y = obs[1]
    angle = obs[2]
    vx = obs[7]
    vy = obs[8]
    current_throttle = obs[5]
    current_angle = obs[6]
    
    # Initialize actions
    throttle_action = 6  # Default: Do nothing
    thruster_action = 6    # Default: Do nothing

    landing_descent_rate = 2.8  # Target descent rate for landing
    maximum_ascent_rate = 0.1  # Maximum ascent rate for landing
    
    if vy < landing_descent_rate:
        # If descending too fast, throttle up to slow down
        throttle_action = 2  # Throttle up
    elif vy > maximum_ascent_rate:
        # If ascending or moving up, throttle down to correct
        throttle_action = 3  # Throttle down
    else:
        # Maintain the current throttle if the descent rate is acceptable
        throttle_action = 6  # Do nothing

    # Adjust the gimbal based on the rocket's angle
    angle_error = 0.0 - angle  # Aim for an upright angle
    desired_angle = pid_angle.compute(angle_error, dt)
    
    if desired_angle > current_angle:
        thruster_action = 5  # Gimbal right
    elif desired_angle < current_angle:
        thruster_action = 4  # Gimbal left
    else:
        thruster_action = 6  # Do nothing

    return throttle_action, thruster_action

# Function to get the current values of the state variables
def get_current_state(obs):
    current_state = {
        'x': obs[0],
        'y': obs[1],
        'angle': obs[2],
        'leg1_contact': obs[3],
        'leg2_contact': obs[4],
        'throttle': obs[5],
        'gimbal': obs[6],
        'vx': obs[7],
        'vy': obs[8],
        'v_angle': obs[9]
    }
    return current_state


def fsm_control_loop(env, render=False):
    """
    The main control loop for the FSM model. This function handles the control flow for landing the rocket,
    based on the current state of the environment.

    Parameters:
    - env: The environment in which the rocket is being controlled.
    - render: Boolean flag to render the environment during each step.

    Returns:
    - flight_data: A list of dictionaries containing state variables for each timestep.
    - actions: A list of actions taken at each timestep.
    - done: Boolean flag indicating if the episode is finished.
    """
    obs = env.reset()
    done = False
    flight_data = []  # Store the raw data from each timestep
    actions = []  # Store the actions taken

    while not done:
        dt = 1.0 / env.metadata['video.frames_per_second']

        # Capture the current state at each timestep
        current_state = get_current_state(obs)  # Get state variables (like angle, gimbal, throttle, etc.)

        # Append the raw state variables to flight_data
        flight_data.append({
            'state': current_state,  # Contains variables like angle, gimbal, throttle, etc.
            'obs': obs  # You can add more data here if needed, such as observations or actions
        })

        # Control logic to decide how to maneuver the rocket
        if within_landing_zone(obs, dt):
            # Landing control logic
            throttle_action, thruster_action = land_rocket(obs, dt)

            # Append actions to the actions list
            actions.append(throttle_action)
            actions.append(thruster_action)

            obs, reward, done, info = env.step(throttle_action)
            obs, reward, done, info = env.step(thruster_action)
        else:
            # Main control loop for non-landing zone
            gimbal_action = correct_angle(obs, dt)
            actions.append(gimbal_action)
            obs, reward, done, info = env.step(gimbal_action)

            throttle_action = set_throttle(obs, dt)
            actions.append(throttle_action)
            obs, reward, done, info = env.step(throttle_action)

            # Apply angle correction again for more aggressive control if moving toward landing zone
            if moving_toward_landing_zone(obs, dt):
                gimbal_action = correct_angle(obs, dt)
                actions.append(gimbal_action)
                obs, reward, done, info = env.step(gimbal_action)

        # Render the environment if specified
        if render:
            env.render()

    return flight_data, actions, done
