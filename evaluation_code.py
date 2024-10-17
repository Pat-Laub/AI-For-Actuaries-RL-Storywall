import stable_baselines3.common as sb3_common
import numpy as np
import matplotlib.pyplot as plt
from atari_helper import *
from collections import Counter

VERSION = "Rushing to Fill Holes"

BASE_DIR = f"./Results/{GAME_NAME}/{ALGO_NAME}/{VERSION}"


data = np.load(BASE_DIR + "/evals/evaluations.npz")
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(f"{GAME_NAME} {ALGO_NAME} Evaluation")
ax1.plot(data["timesteps"], data["results"])
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Mean Reward")
ax2.plot(data["timesteps"], data["ep_lengths"])
ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Mean Episode Length")
plt.show()

vec_env = make_atari_wrapped(
    ENV_ID, N_ENVS, N_STACK, CLIP_REWARD, reward_log=True)

vec_env = sb3_common.vec_env.VecVideoRecorder(
    vec_env, BASE_DIR + "/videos", record_video_trigger=lambda x: x == 0, video_length=10_000, name_prefix="recording")

model = ALGO.load(BASE_DIR + "/evals/best_model")

res = sb3_common.evaluation.evaluate_policy(
    model, vec_env, n_eval_episodes=16, deterministic=True)

reward_storage = []
for env in vec_env.get_attr("env"):
    reward_storage += sb3_common.env_util.unwrap_wrapper(
        env, CustomRewardWrapper).get_rewards()

reward_counts = Counter(reward_storage)
reward_props = {k: v / len(reward_storage) for k, v in reward_counts.items()}

np.savez(BASE_DIR + "/evals/reward_logs",
         rewards=reward_storage, reward_counts=reward_counts, reward_props=reward_props)

print(reward_counts)
print(reward_props)
vec_env.close()
