import os
import time
import matplotlib.pyplot as plt
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import EnvManager, sample_batch

# =================== CONFIG ===================
PLOTS_DIR = "outputs/plots"
TABLE_PATH = "outputs/results_summary.csv"
MODEL_IDS = ["meta-llama/Llama-3.2-3B-Instruct"]# ["meta-llama/Llama-3.2-3B-Instruct"]
ENV_IDS = ["BabyAI-Pickup-v0"]
CONTEXT_WINDOWS = [3, 5]
NUM_ENVS = 4
NUM_BATCHES = 5
BATCH_SIZE = 128
REASONING_FLAG = False
os.makedirs(PLOTS_DIR, exist_ok=True)
from transformers import logging
logging.set_verbosity_error()

def safe_filename(s):
    return s.replace("/", "_")

# =================== MAIN LOOP ===================
for model_id in MODEL_IDS:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_kwargs = {
        "max_new_tokens": 20,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.8,
    }

    for env_id in ENV_IDS:
        for context_window in CONTEXT_WINDOWS:
            print(f"Running {model_id} on {env_id} with context window {context_window}...")

            envs = [
                EnvManager(
                    [env_id],
                    invalid_action_penalty=-2,
                    consecutive_invalid_actions_allowed=5,
                ) for _ in range(NUM_ENVS)
            ]

            total_times = []
            total_generate_times = []
            num_episodes_per_batch = []
            successs = []
            rewardss = []
            episode_lengths = []
            num_invalid_actions = []

            for _ in range(NUM_BATCHES):
                start_time = time.time()
                queries, responses, rewards, stats, running_stats = sample_batch(
                    envs, tokenizer, model, generation_kwargs, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    batch_size=BATCH_SIZE, context_window=context_window, reasoning_flag=REASONING_FLAG
                )
                end_time = time.time()
                sample_batch_time = end_time - start_time
                total_times.append(sample_batch_time)
                total_generate_times.append(stats["total_generate_time"])
                num_episodes = len(running_stats['success'][env_id])
                num_episodes_per_batch.append(num_episodes)

                successs.extend(running_stats['success'][env_id])
                rewardss.extend(running_stats['rewards'][env_id])
                episode_lengths.extend(running_stats['episode_lengths'][env_id])
                num_invalid_actions.extend(running_stats['num_invalid_actions'][env_id])

            # # =================== PLOTTING ===================
            # def save_plot(fig, name):
            #     fig.tight_layout()
            #     fig.savefig(os.path.join(PLOTS_DIR, name))
            #     plt.close(fig)

            # safe_model_id = safe_filename(model_id)

            # fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # axes[0].plot(total_times, label='Total Time per Batch')
            # axes[0].plot(total_generate_times, label='Generate Time per Batch')
            # axes[0].set_title("Time per Batch")
            # axes[0].legend()
            # axes[1].plot(num_episodes_per_batch, label='Episodes per Batch')
            # axes[1].set_title("Number of Episodes")
            # axes[1].legend()
            # save_plot(fig, f"times_and_episodes_{safe_model_id}_{env_id}_ctx{context_window}.png")

            # fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # axes[0].plot(successs, label='Success Rate')
            # axes[0].set_title(f"Avg Success: {sum(successs)/len(successs):.2f}")
            # axes[0].legend()
            # axes[1].plot(rewardss, label='Rewards')
            # axes[1].set_title(f"Avg Reward: {sum(rewardss)/len(rewardss):.2f}")
            # axes[1].legend()
            # save_plot(fig, f"success_and_rewards_{safe_model_id}_{env_id}_ctx{context_window}.png")

            # fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # axes[0].plot(episode_lengths, label='Episode Length')
            # axes[0].set_title(f"Avg Length: {sum(episode_lengths)/len(episode_lengths):.2f}")
            # axes[0].legend()
            # axes[1].plot(num_invalid_actions, label='Invalid Actions')
            # axes[1].set_title(f"Avg Invalid: {sum(num_invalid_actions)/len(num_invalid_actions):.2f}")
            # axes[1].legend()
            # save_plot(fig, f"lengths_and_invalids_{safe_model_id}_{env_id}_ctx{context_window}.png")

            # =================== CSV TABLE ===================
            summary_row = {
                "model_id": model_id,
                "env_id": env_id,
                "context_window": context_window,
                "reasoning_flag": REASONING_FLAG,
                "num_episodes": sum(num_episodes_per_batch),
                "avg_success": sum(successs)/len(successs),
                "avg_reward": sum(rewardss)/len(rewardss),
                "avg_length": sum(episode_lengths)/len(episode_lengths),
                "avg_invalid": sum(num_invalid_actions)/len(num_invalid_actions),
                "avg_total_time": sum(total_times)/len(total_times),
                "avg_generate_time": sum(total_generate_times)/len(total_generate_times),
            }

            df = pd.DataFrame([summary_row])
            if os.path.exists(TABLE_PATH):
                df.to_csv(TABLE_PATH, mode='a', index=False, header=False)
            else:
                df.to_csv(TABLE_PATH, mode='w', index=False, header=True)

print("Done.")