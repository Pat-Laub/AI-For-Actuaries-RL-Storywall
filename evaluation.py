import stable_baselines3.common as sb3_common
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from collections import Counter
import sys
import os

if len(sys.argv) < 2:
    sys.exit("Must supply a version to evaluate")
VERSION = sys.argv[1]

if not os.path.exists(f"./Results/{GAME_NAME}/{ALGO_NAME}/{VERSION}"):
    sys.exit(
        f"Version \"{VERSION}\" does not exist in Results/{GAME_NAME}/{ALGO_NAME}")
BASE_DIR = f"./Results/{GAME_NAME}/{ALGO_NAME}/{VERSION}"

# Create plot with evaluation rewards and episode lengths and save to evaluations.png
data = np.load(BASE_DIR + "/evals/evaluations.npz")
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(f"{GAME_NAME} {ALGO_NAME} Evaluation")
ax1.plot(data["timesteps"], data["results"])
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Mean Reward")
ax2.plot(data["timesteps"], data["ep_lengths"])
ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Mean Episode Length")
plt.savefig(BASE_DIR + "/evals/evaluations.png")
plt.show()

# Creates an environment and puts a video recorder on it for later viewing
vec_env = make_atari_wrapped(
    ENV_ID, N_ENVS, N_STACK, CLIP_REWARD, reward_log=True)

vec_env = sb3_common.vec_env.VecVideoRecorder(
    vec_env, BASE_DIR + "/videos", record_video_trigger=lambda x: x == 0, video_length=10_000, name_prefix="recording")

# Loads best saved model
model = ALGO.load(BASE_DIR + "/evals/best_model")

# Evaluates best saved model
res = sb3_common.evaluation.evaluate_policy(
    model, vec_env, n_eval_episodes=N_EVAL_EPS, deterministic=True)

# Retrieves reward logs from CustomRewardWrapper and saves them to evaluation_reward_logs
reward_storage = []
for env in vec_env.get_attr("env"):
    reward_storage += sb3_common.env_util.unwrap_wrapper(
        env, CustomRewardWrapper).get_rewards()

reward_counts = Counter(reward_storage)
reward_props = {k: v / len(reward_storage) for k, v in reward_counts.items()}

np.savez(BASE_DIR + "/evals/evaluation_reward_logs",
         rewards=reward_storage, reward_counts=reward_counts, reward_props=reward_props)

print(f"Mean Reward: {res[0]} +/- {res[1]}")
print(f"Reward Counts: {dict(reward_counts)}")
print(f"Reward Proportions: {reward_props}")
vec_env.close()
