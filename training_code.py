import stable_baselines3.common as sb3_common
from atari_helper import *
import json

# Initialises environment to train on and evaluate on
train_env = make_atari_wrapped(ENV_ID, N_ENVS, N_STACK, CLIP_REWARD)
eval_env = make_atari_wrapped(ENV_ID, N_ENVS, N_STACK, CLIP_REWARD)

save_dir = create_version_dir(GAME_NAME, ALGO_NAME)

# Callbacks for saving model and evaluating model respectively
checkpoint_callback = sb3_common.callbacks.CheckpointCallback(
    save_freq=SAVE_FREQ, save_path=save_dir + "/saves", name_prefix="model")
eval_callback = sb3_common.callbacks.EvalCallback(eval_env, best_model_save_path=save_dir + "/evals",
                                                  log_path=save_dir + "/evals", eval_freq=EVAL_FREQ,
                                                  deterministic=True, render=False)

# Initialise model and start training
model = ALGO(POLICY, train_env, **
             ALGO_SPECIFIC_PARAMS[ALGO_NAME], verbose=1, device="cuda")

with open(save_dir + "/params.json", "w") as f:
    json.dump(ALGO_SPECIFIC_PARAMS[ALGO_NAME], f)

model.learn(total_timesteps=TRAIN_STEPS, callback=[
            checkpoint_callback, eval_callback], progress_bar=True)
model.save(save_dir + "/saves/final_model")
