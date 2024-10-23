from collections import Counter
import gymnasium
import numpy as np
import stable_baselines3
import stable_baselines3.common as sb3_common
import os
from setup import *

# Generate env id from game name
ENV_ID = f"{GAME_NAME}NoFrameskip-v4"

# Retrieves actual algorithm class
ALGO = getattr(stable_baselines3, ALGO_NAME)

# Adjusts save and evaluation frequency to number of environments
SAVE_FREQ /= N_ENVS
EVAL_FREQ /= N_ENVS

# Clips the reward to -1, 0, 1, i.e. any positive reward becomes +1, any negative reward becomes -1, and any zero reward stays zero
CLIP_REWARD = False


def make_env(env_id, seed=0, reward_log=False):
    '''
    Returns a function that initialises a single environment
    '''
    def _init():
        env = gymnasium.make(env_id, render_mode="rgb_array")
        if reward_log:
            env = CustomRewardWrapper(env)
        env.metadata["render_fps"] = 60
        return env
    sb3_common.utils.set_random_seed(seed)
    return _init


def make_atari_wrapped(env_id, n_envs, n_stack, clip_reward, reward_log=False, seed=0):
    '''
    Wrapper for make_atari_env that also stacks frames and transposes the image
    '''
    env = sb3_common.env_util.make_atari_env(make_env(
        env_id, seed, reward_log), n_envs=n_envs, wrapper_kwargs={"clip_reward": clip_reward})
    env = sb3_common.vec_env.VecFrameStack(env, n_stack=n_stack)
    env = sb3_common.vec_env.VecTransposeImage(env)
    return env


def create_version_dir(game_name, algo_name):
    '''
    Creates a new version directory to save the new model in
    '''
    vers_num = 1
    while True:
        if os.path.exists(f"./Results/{game_name}/{algo_name}/V{vers_num}"):
            vers_num += 1
        else:
            os.makedirs(f"./Results/{game_name}/{algo_name}/V{vers_num}")
            return f"./Results/{game_name}/{algo_name}/V{vers_num}"


class CustomRewardWrapper(gymnasium.RewardWrapper):
    '''
    Wrapper for rewards of environment which tracks the received rewards
    '''

    def __init__(self, env):
        super().__init__(env)
        # Optionally, define the reward range if it changes
        self.reward_storage = []
        self.reward_range = (-float('inf'), float('inf'))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward, terminated = self.reward(reward, terminated)
        return observation, reward, terminated, truncated, info

    def reward(self, reward, terminated):
        # Modify the reward as needed
        self.reward_storage.append(reward)
        return reward, terminated

    def get_rewards(self):
        return self.reward_storage

    def reset_reward_logs(self):
        self.reward_storage = []


class RewardLogCallback(sb3_common.callbacks.BaseCallback):
    '''
    Callback which uses above reward wrapper and then saves the rewards from evaluations in a file
    '''

    def __init__(self, log_path):
        super(RewardLogCallback, self).__init__()
        self.eval_logs = []
        self.log_path = log_path

    def _on_step(self):
        reward_storage = []
        for env in self.parent.eval_env.get_attr("env"):
            reward_env = sb3_common.env_util.unwrap_wrapper(
                env, CustomRewardWrapper)
            reward_storage += reward_env.get_rewards()
            reward_env.reset_reward_logs()

        reward_counts = Counter(reward_storage)
        reward_props = {k: v / len(reward_storage)
                        for k, v in reward_counts.items()}
        self.eval_logs.append({self.num_timesteps: {
                              "full_log": reward_storage, "counts": reward_counts, "props": reward_props}})

        np.savez(self.log_path + "/learning_reward_logs",
                 reward_logs=self.eval_logs)

        print(f"Reward Counts: {reward_counts}")
        print(f"Reward Proportions: {reward_props}")
        return True
