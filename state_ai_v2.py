import numpy as np
import gym
from gym.envs.registration import register

# Register the custom environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Assuming rocket_lander.py is in the same directory
    max_episode_steps=1000,
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

class RocketLanderFSM:
    def __init__(self):
        self.state = "FLYING"
        self.pid_x = PIDController(10.0, 0.0, 8.0, output_limits=(-1.0, 1.0))  # Further fine-tuned lateral control
        self.pid_y = PIDController(100.0, 0.0, 800.0, output_limits=(-1.0, 1.0))  # Adjusted vertical control
        self.pid_angle = PIDController(1200.0, 0.0, 800.0, output_limits=(-1.0, 1.0))  # Stronger gimbal control

    def set_state(self, state, dt):
        x, y, angle, leg1_contact, leg2_contact, throttle, gimbal, vx, vy, v_angle = state

        # Adjusting the condition to transition to LANDING state sooner
        if self.state == "FLYING":
            if y != "1":
                self.state = "LANDING"

    def set_throttle(self, state, dt, env):
        x, y, angle, leg1_contact, leg2_contact, throttle, gimbal, vx, vy, v_angle = state

        if self.state == "LANDING":
            action = np.zeros(3)
            discrete_action = 6

            # Target vertical velocity should start high and decrease as altitude decreases
            # The closer the rocket is to the ground, the closer target_vy is to 0
            max_descent_speed = -10.0  # Maximum descent speed when high above the ground
            min_descent_speed = -0.5   # Minimum descent speed when close to the ground
            target_vy = max_descent_speed * y + min_descent_speed * (1 - y)

            # Immediate response to velocity error
            throttle_error = target_vy - vy
            action[1] = self.pid_y.compute(throttle_error, dt)

            # Adjust the thrust proportionally based on the throttle error
            if action[1] > 0.001:  # Throttle up
                discrete_action = 2
            elif action[1] < -0.001:  # Throttle down
                discrete_action = 3

            return discrete_action

    def correct_angle_gimbal(self, state, dt, env):
        x, y, angle, leg1_contact, leg2_contact, throttle, gimbal, vx, vy, v_angle = state

        if self.state == "LANDING":
            actions = np.zeros(1)  # Initialize an array to store the gimbal action

            # Immediate response to angle error
            angle_error = 0.0 - angle  # We want the angle to be 0 (upright)
            desired_gimbal = self.pid_angle.compute(angle_error, dt)

            # Clip the desired gimbal angle to be within the valid range (e.g., -1 to 1)
            desired_gimbal = np.clip(desired_gimbal, -1.0, 1.0)

            # # Assign the gimbal action
            if desired_gimbal > 0:
                actions[0] = 1  # Gimbal right
            elif desired_gimbal < 0:
                actions[0] = 0  # Gimbal left

            return actions

    def move_laterally_thrusters(self, state, dt, env):
        x, y, angle, leg1_contact, leg2_contact, throttle, gimbal, vx, vy, v_angle = state

        if self.state == "LANDING":
            action = np.zeros(3)
            discrete_action = 6

            # Immediate response to position error and velocity
            position_error = 0.0 - x
            velocity_error = 0.0 - vx  # We want the horizontal velocity to be zero

            # Combine position and velocity errors for smoother control
            combined_error = position_error + velocity_error
            action[2] = self.pid_x.compute(combined_error, dt)

            if action[2] > 0.1:
                discrete_action = 4  # Apply thrust to the left
            elif action[2] < -0.1:
                discrete_action = 5  # Apply thrust to the right

            return discrete_action

# Run a single episode with the FSM-controlled rocket
obs = env.reset()
fsm = RocketLanderFSM()
done = False
step = 0
while not done:
    dt = 1.0 / env.metadata['video.frames_per_second']
    state = fsm.set_state(obs, dt)

    action = fsm.correct_angle_gimbal(obs, dt, env)
    obs, reward, done, info = env.step(action)

    action= fsm.set_throttle(obs, dt, env)
    obs, reward, done, info = env.step(action)
    
    action = fsm.move_laterally_thrusters(obs, dt, env)
    obs, reward, done, info = env.step(action)

    #make the print statment appear every 100 steps
    # if step % 100 == 0:
    #     print(f"State: {fsm.state}, Position: ({obs[0]:.2f}, {obs[1]:.2f}), Angle: {obs[2]:.2f}, "
    #           f"Velocity: ({obs[7]:.2f}, {obs[8]:.2f}), Angular Velocity: {obs[9]:.2f}, "
    #           f"Throttle: {obs[5]:.2f}, Gimbal: {obs[6]:.2f}")
    step += 1 
    env.render()
