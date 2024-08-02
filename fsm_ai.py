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

    def reset(self):
        self.state = "FLYING"
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_angle.reset()

    def get_action(self, state, dt):
        x, y, angle, leg1_contact, leg2_contact, throttle, gimbal, vx, vy, v_angle = state

        # Adjusting the condition to transition to LANDING state sooner
        if self.state == "FLYING":
            if y != "1":
                self.state = "LANDING"

        if self.state == "LANDING":
            action = np.zeros(3)

            # Immediate response to velocity error
            throttle_error = 1.0 - vy
            action[1] = self.pid_y.compute(throttle_error, dt)

            gimbal_error = 0 - angle
            action[0] = self.pid_angle.compute(gimbal_error, dt)

            if vx > 0.05:
                action[2] = -0.5
            elif vx < -0.05:
                action[2] = 0.5

            # Convert continuous actions to discrete actions
            discrete_action = 6  # Default action (no action)
            if action[0] > 0.001:  # Lower threshold for gimbal left
                discrete_action = 0  # Gimbal left
            elif action[0] < -0.001:  # Lower threshold for gimbal right
                discrete_action = 1  # Gimbal right
            if action[1] > 0.001:  # Lower threshold for throttle up
                discrete_action = 2  # Throttle up
            elif action[1] < -0.8:  # Lower threshold for throttle down
                discrete_action = 3  # Throttle down
            if action[2] > 0.001:  # Lower threshold for control thruster left
                discrete_action = 4  # Control thruster left
            elif action[2] < -0.001:  # Lower threshold for control thruster right
                discrete_action = 5  # Control thruster right



            return discrete_action

        return 6  # Default action (no action)

# Run a single episode with the FSM-controlled rocket
obs = env.reset()
fsm = RocketLanderFSM()
done = False
step = 0
while not done:
    dt = 1.0 / env.metadata['video.frames_per_second']
    action = fsm.get_action(obs, dt)
    obs, reward, done, info = env.step(action)
    
    #make the print statment appear every 100 steps
    if step % 100 == 0:
        print(f"State: {fsm.state}, Position: ({obs[0]:.2f}, {obs[1]:.2f}), Angle: {obs[2]:.2f}, "
              f"Velocity: ({obs[7]:.2f}, {obs[8]:.2f}), Angular Velocity: {obs[9]:.2f}, "
              f"Throttle: {obs[5]:.2f}, Gimbal: {obs[6]:.2f}")
    step += 1 
    env.render()
