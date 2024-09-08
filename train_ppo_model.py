import gym
import tensorflow as tf
import os
import datetime
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from gym.envs.registration import register
from stable_baselines.common.callbacks import BaseCallback
import zipfile
import shutil

# Register the RocketLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',  # Adjust the path as needed
    max_episode_steps=3000,
    reward_threshold=0,
)

# Defining utils
n_cpu = 4  # Number of CPU cores to use for training
timestep = 20000000
ENV = 'RocketLander-v0'
timestamp = datetime.datetime.now()
filename = "ppo2_{}_{}".format(ENV, str(timestamp)[:19])

# Create log dir
path = '{}_tensorboard'.format(ENV[:-3])
if not os.path.exists(path):
    os.makedirs(path)

# Ensure model directory exists
if not os.path.exists('./model/'):
    os.makedirs('./model/')

# Function to initialize each environment instance
def make_env():
    def _init():
        env = gym.make('RocketLander-v0')
        env = Monitor(env, 'Monitor_Log', allow_early_resets=True)
        return env
    return _init

MAX_LOG_SIZE = 10 * 1024 * 1024 * 1024  # 10GB

# Use DummyVecEnv for debugging
env = DummyVecEnv([make_env() for _ in range(n_cpu)])

# Function to get the total size of the log directory
def get_log_dir_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# Function to manage log size by compressing older log files
def manage_log_size(directory, max_size):
    while get_log_dir_size(directory) > max_size:
        log_files = sorted(
            (os.path.join(dirpath, f) for dirpath, dirnames, filenames in os.walk(directory) for f in filenames),
            key=os.path.getctime
        )
        if log_files:
            log_file = log_files[0]
            compressed_log_file = log_file + '.zip'
            with zipfile.ZipFile(compressed_log_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(log_file, os.path.basename(log_file))
            os.remove(log_file)
            print(f"Compressed and removed log file: {log_file}")
        else:
            break

# Custom callback to handle logging and model saving
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.model_save_counter = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 == 0:
            print(f"Timestep: {self.num_timesteps}/{timestep}")

        # Save the model at evenly spaced intervals
        if self.num_timesteps % (timestep // 10) == 0:
            save_path = f'./model/{filename}_step_{self.num_timesteps}'
            self.model.save(save_path)
            self.model_save_counter += 1
            print(f'Model saved at timestep {self.num_timesteps} to {save_path}')

            # Ensure we only keep the last 10 models
            if self.model_save_counter > 10:
                oldest_model_index = self.model_save_counter - 10
                oldest_model_path = f'./model/{filename}_step_{oldest_model_index * (timestep // 10)}'
                if os.path.exists(oldest_model_path):
                    os.remove(oldest_model_path)
                    print(f'Removed old model: {oldest_model_path}')
        return True

    def _on_training_start(self) -> None:
        # Check and manage log size
        manage_log_size(path, MAX_LOG_SIZE)
        manage_log_size('Monitor_Log', MAX_LOG_SIZE)

    def _on_training_end(self) -> None:
        # Check and manage log size
        manage_log_size(path, MAX_LOG_SIZE)
        manage_log_size('Monitor_Log', MAX_LOG_SIZE)

config = tf.ConfigProto()
# if GPU uncomment below couple of lines of code
# config = tf.ConfigProto(device_count={'GPU': 0})
# config.gpu_options.allow_growth = True

# Let's run a tensorflow session
with tf.Session(config=config):
    model = PPO2(MlpPolicy, env, n_steps=1024, nminibatches=256, lam=0.95, gamma=0.99, noptepochs=3, ent_coef=0.01, learning_rate=lambda _: 1e-4, cliprange=lambda _: 0.2, tensorboard_log=path, full_tensorboard_log=True, verbose=2)
    model.learn(total_timesteps=timestep, log_interval=1000, callback=CustomCallback())  # 15M timesteps and overnight run on a Macbook worked fine (still can improve).

    final_model_path = './model/' + filename
    model.save(final_model_path)
    print(f'Final Model Saved at {final_model_path}')

    # Manage logs if they exceed the size limit
    manage_log_size(path, MAX_LOG_SIZE)
    manage_log_size('Monitor_Log', MAX_LOG_SIZE)
