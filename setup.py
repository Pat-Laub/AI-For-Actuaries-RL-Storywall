# Which game from Atari to train on - https://gymnasium.farama.org/environments/atari/
# Note you can only pick games which have a NoFrameskip-v4 version
GAME_NAME = "Gopher"

# Name of algorithm to use from SB3 (PPO, A2C, DQN, etc.)
ALGO_NAME = "DQN"

# How many actions the agent takes to train
TRAIN_STEPS = 10_000_000

# How many episodes to evaluate the model on
N_EVAL_EPS = 8

# How often to save the model - note this number is multiplied by N_ENVS to get the actual amount of timesteps that pass before saving
SAVE_FREQ = 500_000

# How often to evaluate the model - same note as above
EVAL_FREQ = 1_000_000

# Discount rate for rewards received
DISCOUNT_RATE = 0.99


# ---------------------------- You shouldn't need to change anything below here ----------------------------


# How many environments to run in parallel
N_ENVS = 8

# How many frames to stack together
# See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
N_STACK = 4

# Which policy to use for the model
POLICY = "CnnPolicy"


# Specific parameters for each algorithm - taken from RL-Zoo3
# They're well documented in the SB3 documentation i.e. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO
# with values taken from https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
ALGO_SPECIFIC_PARAMS = {
    "PPO": {
        "n_epochs": 4,
        "batch_size": 1024,
        "learning_rate": 0.0003,
        "clip_range": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "n_steps": 256,
        "clip_range_vf": None,
        "gamma": DISCOUNT_RATE
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
        "gamma": DISCOUNT_RATE,

    }
}
