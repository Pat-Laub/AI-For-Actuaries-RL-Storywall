import stable_baselines3.common as sb3_common
from helper import *


# Which game from Atari to train on - https://gymnasium.farama.org/environments/atari/
GAME_NAME = "Gopher"
ENV_ID = f"{GAME_NAME}NoFrameskip-v4"

# Clips the reward to -1, 0, 1, i.e. any positive reward becomes +1, any negative reward becomes -1, and any zero reward stays zero
CLIP_REWARD = False

# How many environments to run in parallel
N_ENVS = 8
SAVE_FREQ /= N_ENVS
EVAL_FREQ /= N_ENVS

# How many frames to stack together
N_STACK = 4

# Which policy to use for the model
POLICY = "CnnPolicy"

# Specific parameters for each algorithm
# They're well documented in the SB3 documentation i.e. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO
ALGO_SPECIFIC_PARAMS = {
    "PPO": {
        "n_epochs": 4,
        "batch_size": 1024,
        "learning_rate": 0.0003,
        "clip_range": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "n_steps": 256,
        "clip_range_vf": None
    },
    "DQN": {
        "buffer_size": 100_000,
        "learning_rate": 0.00008,
        "batch_size": 64,
        "learning_starts": 100_000,
        "target_update_interval": 1_000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "optimize_memory_usage": False,
        "gamma": 0.5,

    }
}


def make_atari_wrapped(env_id, n_envs, n_stack, clip_reward, reward_log=False, seed=0):
    '''
    Wrapper for make_atari_env that also stacks frames and transposes the image
    '''
    env = sb3_common.env_util.make_atari_env(make_env(
        env_id, seed, reward_log), n_envs=n_envs, wrapper_kwargs={"clip_reward": clip_reward})
    env = sb3_common.vec_env.VecFrameStack(env, n_stack=n_stack)
    env = sb3_common.vec_env.VecTransposeImage(env)
    return env
