import os
import matplotlib.pyplot as plt
from fsm_controller import env, get_current_state, within_landing_zone, land_rocket, correct_angle, set_throttle, moving_toward_landing_zone

# Create the directory if it doesn't exist
os.makedirs('plots/FSM', exist_ok=True)

# Initialize lists to store the state variables for plotting
x_vals = []
y_vals = []
angle_vals = []
throttle_vals = []
gimbal_vals = []
vx_vals = []
vy_vals = []

# Run a single episode with the FSM-controlled rocket
obs = env.reset()
done = False
while not done:
    dt = 1.0 / env.metadata['video.frames_per_second']
    current_state = get_current_state(obs)

    # Record the state variables
    x_vals.append(obs[0])
    y_vals.append(obs[1])
    angle_vals.append(obs[2])
    throttle_vals.append(obs[5])
    gimbal_vals.append(obs[6])
    vx_vals.append(obs[7])
    vy_vals.append(obs[8])

    if within_landing_zone(obs, dt):
        # Landing control loop
        throttle_action, thruster_action = land_rocket(obs, dt)
        obs, reward, done, info = env.step(throttle_action)
        obs, reward, done, info = env.step(thruster_action)
    else:
        # Main control loop
        current_state = get_current_state(obs)
        action = correct_angle(obs, dt)
        obs, reward, done, info = env.step(action)

        current_state = get_current_state(obs)
        action = set_throttle(obs, dt)
        obs, reward, done, info = env.step(action)

        # Apply angle correction again for more aggressive angle control if moving toward landing zone
        if moving_toward_landing_zone(obs, dt):
            current_state= get_current_state(obs)
            action = correct_angle(obs, dt)
            obs, reward, done, info = env.step(action)

# Plot and save the graphs
plt.figure()
plt.plot(x_vals, y_vals)
plt.title('Rocket Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.savefig('plots/FSM/trajectory.png')

plt.figure()
plt.plot(range(len(angle_vals)), angle_vals)
plt.title('Rocket Angle Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Angle (Radians)')
plt.grid(True)
plt.savefig('plots/FSM/angle_correction.png')

plt.figure()
plt.plot(range(len(throttle_vals)), throttle_vals)
plt.title('Throttle Adjustment Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Throttle')
plt.grid(True)
plt.savefig('plots/FSM/throttle_adjustments.png')

plt.figure()
plt.plot(range(len(gimbal_vals)), gimbal_vals)
plt.title('Gimbal Angle Adjustments Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Gimbal Angle (Radians)')
plt.grid(True)
plt.savefig('plots/FSM/gimbal_angle_adjustments.png')

# plt.figure()
# plt.plot(range(len(vx_vals)), vx_vals, label='Vx (Horizontal Velocity)')
# plt.plot(range(len(vy_vals)), vy_vals, label='Vy (Vertical Velocity)')
# plt.title('Horizontal and Vertical Velocities Over Time')
# plt.xlabel('Time Steps')
# plt.ylabel('Velocity')
# plt.legend()
# plt.grid(True)
# plt.savefig('plots/FSM/velocities.png')

plt.figure()
adjusted_vy_vals = [vy - 3 for vy in vy_vals]
plt.plot(range(len(vx_vals)), vx_vals, label='Vx (Horizontal Velocity)')
plt.plot(range(len(adjusted_vy_vals)), adjusted_vy_vals, label='Vy (Vertical Velocity)')
plt.title('Horizontal and Vertical Velocities Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.savefig('plots/FSM/velocities.png')

plt.close('all')

print("Plots saved in the 'plots/FSM/' directory.")
