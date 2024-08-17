import numpy as np
import gym
from gym.envs.registration import register

# Register the custom environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Assuming rocket_lander.py is in the same directory
    max_episode_steps=1900,
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
pid_vx = PIDController(15.0, 0.0, 5.0, output_limits=(-1.0, 1.0))  # Increase the gain for more aggressive control

def correct_angle(current_angle, current_gimbal, current_x, current_vx, dt):
    # Set limits for x position and velocity
    position_tolerance = 0.01  # Acceptable range for x position
    velocity_tolerance = 0.1  # Acceptable range for horizontal velocity

    # Calculate the target angle based on current x position and horizontal velocity
    if abs(current_x) > position_tolerance or abs(current_vx) > velocity_tolerance:
        # If x position or velocity is outside the limits, allow a maximum tilt
        max_angle = np.clip(current_x * 0.1 + current_vx * 0.2, -0.3, 0.3)  # Dynamically determine the max angle
    else:
        # If within limits, aim for an upright angle
        max_angle = 0.0

    # Calculate the error based on the dynamically determined target angle
    angle_error = max_angle - current_angle
    desired_gimbal = pid_angle.compute(angle_error, dt)

    # Set the gimbal action based on the desired angle correction
    if desired_gimbal > current_gimbal:
        return 1  # Gimbal right
    elif desired_gimbal < current_gimbal:
        return 0  # Gimbal left
    else:
        return 6  # Do nothing

def set_throttle(current_vy, current_throttle, current_vx, current_x,dt):

    target_vy = 3.1

    if abs(current_vx) >= 0.1 and current_vy < 0.1:
        target_vy = 7
    if abs(current_x) > 0 and current_vy < 0.1:
        target_vy = 9
    if abs(current_x) < 0 and current_vy < 0.1:
        target_vy = 9

    vertical_velocity_error = target_vy - current_vy

    desired_throttle = pid_throttle.compute(vertical_velocity_error, dt)

    if current_throttle < desired_throttle:
        return 2  # Throttle up
    elif current_throttle > desired_throttle:
        return 3  # Throttle down
    else:
        return 6  # Do nothing

    # Apply the computed throttle
    if current_throttle < desired_throttle:
        return 2  # Throttle up
    elif current_throttle > desired_throttle:
        return 3  # Throttle down
    else:
        return 6  # Do nothing

def kill_horizontal_velocity(current_x, current_vx, dt):
    action = 6

    if current_vx > 0.1:
        action = 1
    elif current_vx < 0.1:
        action = 0

    return action

def move_towards_landing_pad(current_x):
    if current_x > 0:
        return 1  # Move left
    elif current_x < 0:
        return 0  # Move right<
    else:
        return 6  # Do nothing

def get_current_state(obs):
    """
    Extracts and returns the current state values from the observation.

    Parameters:
    - obs: The observation returned by the environment after applying an action.

    Returns:
    - A dictionary containing the current state values.
    """
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

start_vel = 0 

# Run a single episode with the FSM-controlled rocket
obs = env.reset()
done = False
step = 0

while not done:
    dt = 1.0 / env.metadata['video.frames_per_second']

    # Control loop
    current_state = get_current_state(obs)
    
    if step == 0:
        start_vel = obs[7]

    action = correct_angle(obs[2], obs[6], obs[0], obs[7], dt)
    obs, reward, done, info = env.step(action)

    current_state = get_current_state(obs)
    action = set_throttle(obs[8], obs[5], obs[7], obs[0], dt)
    obs, reward, done, info = env.step(action)

    current_state = get_current_state(obs)
    action = kill_horizontal_velocity(obs[0], obs[7], dt)
    obs, reward, done, info = env.step(action)

    current_state = get_current_state(obs)
    action = correct_angle(obs[2], obs[6], obs[0], obs[7], dt)
    obs, reward, done, info = env.step(action)

    if abs(obs[7]) <= 0.05 or abs(obs[0]) >= 0:
        current_state = get_current_state(obs)
        action = move_towards_landing_pad(obs[0])
        obs, reward, done, info = env.step(action)

        current_state = get_current_state(obs)
        action = correct_angle(obs[2], obs[6], obs[0], obs[7], dt)
        obs, reward, done, info = env.step(action)
     
    step += 1
    env.render()

print(start_vel)
