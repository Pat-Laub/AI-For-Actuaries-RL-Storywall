import gymnasium
import stable_baselines3
import stable_baselines3.common as sb3_common
import os

# How many actions the agent takes to train
TRAIN_STEPS = 10_000_000

# Name of algorithm to use from SB3 (PPO, A2C, DQN, etc.)
ALGO_NAME = "DQN"
ALGO = getattr(stable_baselines3, ALGO_NAME)

# How many episodes to evaluate the model on
N_EVAL_EPS = 5

# How often to save the model - note this number is multiplied by N_ENVS to get the actual amount of timesteps that pass before saving
SAVE_FREQ = 500_000

# How often to evaluate the model - same note as above
EVAL_FREQ = 250_000


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


def create_version_dir(game_name, algo_name):
    '''
    Returns the directory to save the model in
    '''
    vers_num = 1
    while True:
        if os.path.exists(f"./Results/{game_name}/{algo_name}/V{vers_num}"):
            vers_num += 1
        else:
            os.makedirs(f"./Results/{game_name}/{algo_name}/V{vers_num}")
            return f"./{game_name}/{algo_name}/V{vers_num}"


class CustomRewardWrapper(gymnasium.RewardWrapper):
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
