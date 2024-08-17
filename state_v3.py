import numpy as np
import gym
from gym.envs.registration import register

# Register the custom environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Assuming rocket_lander.py is in the same directory
    max_episode_steps=1200,
    reward_threshold=0,
)

def correct_angle(current_angle):
    # set vehicle at the correct angle
    angle_error = 0.0 - current_angle # We want the angle to be 0 (upright)
    desired_gimbal = pid_angle.compute(angle_error, dt)

    # Assign the gimbal action
    if desired_gimbal > current_gimbal:
        action = 1  # Gimbal right
    elif desired_gimbal < current_gimbal:
        action = 0  # Gimbal left
    else:
        action = 6 # Do nothing
    return action

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

pid_angle = PIDController(1200.0, 0.0, 800.0, output_limits=(-3.5, 3.5))  # Stronger gimbal control
pid_throttle = PIDController(400.0, 0.0, 200.0, output_limits=(-1.27,0.96))  # Stronger gimbal control

# Run a single episode with the FSM-controlled rocket
obs = env.reset()
done = False
step = 0
while not done:
    dt = 1.0 / env.metadata['video.frames_per_second']

    # default action is to do nothing
    action = 6

    current_gimbal = obs[6]
    current_throttle = obs[5]
    current_x = obs[0]
    current_y = obs[1]
    current_angle = obs[2]
    current_vx = obs[7]
    current_vy = obs[8]
    current_v_angle = obs[9]
    current_leg1_contact = obs[3]
    current_leg2_contact = obs[4]

    action = correct_angle(current_angle)
    obs, reward, done, info = env.step(action)

    # set throttle
    vertical_velocity_error = 2.9 - current_vy

    desired_throttle = pid_throttle.compute(vertical_velocity_error, dt)

    obs, reward, done, info = env.step(action)

    # Assign the throttle action
    if current_throttle < desired_throttle:
        action = 2
    elif current_throttle > desired_throttle:
        action = 3

    obs, reward, done, info = env.step(action)

    # kill horizontal velocity
    if current_vx > 0:
        action = 1
    elif current_vx < 0:
        action = 0

    obs, reward, done, info = env.step(action)

    action = correct_angle(current_angle)
    obs, reward, done, info = env.step(action)

    # maneuver to the landing pad
    if current_vx <= 0.1 or current_vx >= -0.1:
        
        if current_x > 0.0:
            action = 1
        elif current_x < 0.0:
            action = 0
        #prevent angle from exceeding 0.1 radians to prevent inducing too much horizontal velocity
        if current_angle >= 0.1:
            action = 1
        elif current_angle <= -0.1:
            action = 0

    obs, reward, done, info = env.step(action)

    step += 1 
    env.render()
