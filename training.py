from collections import Counter
import numpy as np
import stable_baselines3.common as sb3_common
from helper import *
import json

# Initialises environment to train on and evaluate on
train_env = make_atari_wrapped(ENV_ID, N_ENVS, N_STACK, CLIP_REWARD)
eval_env = make_atari_wrapped(
    ENV_ID, N_ENVS, N_STACK, CLIP_REWARD, reward_log=True)

# Creates a new directory to save evaluations and checkpoints in
save_dir = create_version_dir(GAME_NAME, ALGO_NAME)

# Callback which backs up the model every SAVE_FREQ steps
checkpoint_callback = sb3_common.callbacks.CheckpointCallback(
    save_freq=SAVE_FREQ, save_path=save_dir + "/saves", name_prefix="model")

# Pair of callbacks which evaluate the model every EVAL_FREQ steps
# and track all rewards received in the evaluations
reward_log_callback = RewardLogCallback(save_dir + "/evals")
eval_callback = sb3_common.callbacks.EvalCallback(eval_env, best_model_save_path=save_dir + "/evals",
                                                  log_path=save_dir + "/evals", eval_freq=EVAL_FREQ,
                                                  deterministic=True, render=False, callback_after_eval=reward_log_callback)


# Initialise model
model = ALGO(POLICY, train_env, **
             ALGO_SPECIFIC_PARAMS[ALGO_NAME], verbose=1, device="cuda")

# Dumps parameters to params.json
with open(save_dir + "/params.json", "w") as f:
    json.dump(ALGO_SPECIFIC_PARAMS[ALGO_NAME], f)

# Trains model and saves at the end
model.learn(total_timesteps=TRAIN_STEPS, callback=[
            checkpoint_callback, eval_callback], progress_bar=True)
model.save(save_dir + "/saves/final_model")
