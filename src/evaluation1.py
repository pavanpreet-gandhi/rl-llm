import numpy as np
import torch
import gym
import time
import matplotlib.pyplot as plt
from env_manager import EnvManager
import babyai_text
import utils
from peft import PeftModel, PeftConfig
from typing import Dict, Any, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from src import sample_batch
import logging
import os

# Suppress transformers logging to ERROR level
logging.getLogger("transformers").setLevel(logging.ERROR)

# Create directories for outputs if they don’t exist
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

babyai_text.register_levels(__name__, globals())

# Evaluation function
def evaluate(
    model: torch.nn.Module,
    tokenizer,
    env_id: str,
    context_window: int,
    num_envs: int = 4,
    num_batches: int = 10,
    batch_size: int = 128,
    reasoning_flag: bool = False,
    generation_kwargs: dict = None,
    device: torch.device = None,
    invalid_action_penalty: float = -2,
    consecutive_invalid_actions_allowed: int = 5,
    model_name: str = "model",
    step_offset: int = 0,
) -> Dict[str, Any]:
    print(f"Evaluating {env_id} with {num_batches} batches...")

    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.95,
            "temperature": 0.8,
            "pad_token_id": tokenizer.eos_token_id,
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        envs = [
            EnvManager(
                [env_id],
                invalid_action_penalty=invalid_action_penalty,
                consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
            ) for _ in range(num_envs)
        ]
    except Exception as e:
        print(f"Failed to initialize environments for {env_id}: {e}")
        raise

    total_times = []
    total_generate_times = []
    num_episodes_per_batch = []
    successs = []
    rewardss = []
    episode_lengths = []
    num_invalid_actions = []
    running_success_rates = []
    running_avg_rewards = []
    batch_times = []

    running_num_episodes = 0
    running_success_sum = 0
    running_reward_sum = 0
    running_length_sum = 0
    running_invalid_sum = 0
    running_success_lengths = []

    eval_start_time = time.time()

    for batch_idx in range(num_batches):
        start_time = time.time()

        # Temporarily set transformers logging to ERROR to suppress warnings
        logger = logging.getLogger("transformers")
        original_level = logger.level
        logger.setLevel(logging.ERROR)

        _, _, rewards, stats, running_stats = sample_batch(
            envs=envs,
            tokenizer=tokenizer,
            model=model,
            generation_kwargs=generation_kwargs,
            device=device,
            batch_size=batch_size,
            context_window=context_window,
            reasoning_flag=reasoning_flag
        )

        # Restore the original logging level
        logger.setLevel(original_level)

        end_time = time.time()
        batch_time = end_time - start_time
        elapsed_time = end_time - eval_start_time
        batch_times.append(elapsed_time)

        total_times.append(batch_time)
        total_generate_times.append(stats["total_generate_time"])
        num_episodes = len(running_stats['success'][env_id])
        num_episodes_per_batch.append(num_episodes)

        batch_success = running_stats['success'][env_id]
        batch_rewards = running_stats['rewards'][env_id]
        batch_lengths = running_stats['episode_lengths'][env_id]
        batch_invalids = running_stats['num_invalid_actions'][env_id]

        successs.extend(batch_success)
        rewardss.extend(batch_rewards)
        episode_lengths.extend(batch_lengths)
        num_invalid_actions.extend(batch_invalids)

        running_num_episodes += num_episodes
        running_success_sum += sum(batch_success)
        running_reward_sum += sum(batch_rewards)
        running_length_sum += sum(batch_lengths)
        running_invalid_sum += sum(batch_invalids)
        running_success_lengths.extend([length for length, success in zip(batch_lengths, batch_success) if success])

        num_episodes_so_far = len(successs)
        running_success_rate = running_success_sum / num_episodes_so_far if num_episodes_so_far > 0 else 0
        running_success_rates.append(running_success_rate)

        total_rewards_so_far = sum(rewardss)
        running_avg_reward = total_rewards_so_far / num_episodes_so_far if num_episodes_so_far > 0 else 0
        running_avg_rewards.append(running_avg_reward)

    # Compute standard deviations for the metrics
    metrics = {
        "num_episodes": sum(num_episodes_per_batch),
        "success_rate": sum(successs) / len(successs) if successs else 0,
        "success_rate_std": np.std(successs) if successs else 0,
        "avg_reward": sum(rewardss) / len(rewardss) if rewardss else 0,
        "avg_reward_std": np.std(rewardss) if rewardss else 0,
        "avg_steps": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
        "avg_steps_std": np.std(episode_lengths) if episode_lengths else 0,
        "invalid_action_rate": sum(num_invalid_actions) / len(num_invalid_actions) if num_invalid_actions else 0,
        "invalid_action_rate_std": np.std(num_invalid_actions) if num_invalid_actions else 0,
        "avg_steps_to_success": (
            sum([length for length, success in zip(episode_lengths, successs) if success]) /
            sum(successs) if sum(successs) > 0 else float("nan")
        ),
        "avg_steps_to_success_std": np.std(running_success_lengths) if running_success_lengths else float("nan"),
        "avg_total_time": sum(total_times) / len(total_times),
        "avg_total_time_std": np.std(total_times) if total_times else 0,
        "avg_generate_time": sum(total_generate_times) / len(total_generate_times),
        "avg_generate_time_std": np.std(total_generate_times) if total_generate_times else 0,
    }

    return {
        "final_metrics": metrics,
        "running_success_rates": running_success_rates,
        "running_avg_rewards": running_avg_rewards,
        "batch_times": batch_times,
    }

# Plotting function to create a bar chart for a single checkpoint
def plot_performance(
    env_ids: List[str],
    context_window: int,
    baseline_results: Dict[str, float],
    reasoning_results: Dict[str, float],
    non_reasoning_results: Dict[str, float],
    env_type: str,
    checkpoint_name: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_envs = len(env_ids)
    bar_width = 0.25
    index = np.arange(n_envs)

    ax.bar(index - bar_width, [baseline_results[env_id] for env_id in env_ids], 
           bar_width, label="Baseline (Untrained)", color='red', alpha=0.7)
    ax.bar(index, [reasoning_results[env_id] for env_id in env_ids], 
           bar_width, label="Reasoning (Trained)", color='blue', alpha=0.7)
    ax.bar(index + bar_width, [non_reasoning_results[env_id] for env_id in env_ids], 
           bar_width, label="Non-Reasoning (Trained)", color='green', alpha=0.7)

    ax.set_xlabel("Environments", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(f"Model Performance Comparison (Checkpoint: {checkpoint_name}, Context Window: {context_window}, {env_type})", fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(env_ids, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = f"outputs/plots/{env_type.lower()}_comparison_{checkpoint_name}_ctx{context_window}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# Evaluate models function
def evaluate_models(
    models_info: List[Dict[str, Any]],
    seen_env_ids: List[str],
    unseen_env_ids: List[str],
    context_windows: List[int] = [1, 2, 3, 4, 5],
    num_envs: int = 4,
    num_batches: int = 3,
    batch_size: int = 128,
    generation_kwargs: dict = None,
    device: torch.device = None,
    invalid_action_penalty: float = -2,
    consecutive_invalid_actions_allowed: int = 5,
    step_offset: int = 0,
) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.95,
            "temperature": 0.8,
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    current_step = step_offset

    # Find the baseline model
    baseline_model_info = None
    for model_info in models_info:
        if "baseline" in model_info['name'].lower():
            baseline_model_info = model_info
            break

    if baseline_model_info is None:
        raise ValueError("Baseline (untrained) model not found in models_info.")

    # Evaluate the baseline model on all environments
    baseline_results = {"seen": {}, "unseen": {}}
    for env_id in seen_env_ids:
        baseline_results["seen"][env_id] = {}
        for context_window in context_windows:
            print(f"\n=== Evaluating baseline model on seen env {env_id} with context {context_window} ===")
            result = evaluate(
                model=baseline_model_info['model'],
                tokenizer=baseline_model_info['tokenizer'],
                env_id=env_id,
                context_window=context_window,
                num_envs=num_envs,
                num_batches=num_batches,
                batch_size=batch_size,
                reasoning_flag=False,
                generation_kwargs=generation_kwargs,
                device=device,
                invalid_action_penalty=invalid_action_penalty,
                consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                model_name=baseline_model_info['name'],
                step_offset=current_step,
            )
            baseline_results["seen"][env_id][context_window] = result["final_metrics"]["success_rate"]
            current_step += num_batches

    for env_id in unseen_env_ids:
        baseline_results["unseen"][env_id] = {}
        for context_window in context_windows:
            print(f"\n=== Evaluating baseline model on unseen env {env_id} with context {context_window} ===")
            result = evaluate(
                model=baseline_model_info['model'],
                tokenizer=baseline_model_info['tokenizer'],
                env_id=env_id,
                context_window=context_window,
                num_envs=num_envs,
                num_batches=num_batches,
                batch_size=batch_size,
                reasoning_flag=False,
                generation_kwargs=generation_kwargs,
                device=device,
                invalid_action_penalty=invalid_action_penalty,
                consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                model_name=baseline_model_info['name'],
                step_offset=current_step,
            )
            baseline_results["unseen"][env_id][context_window] = result["final_metrics"]["success_rate"]
            current_step += num_batches

    # Evaluate each trained model
    for model_info in models_info:
        if "baseline" in model_info['name'].lower():
            continue

        model = model_info['model']
        tokenizer = model_info['tokenizer']
        model_name = model_info['name']
        checkpoint_name = model_info.get('checkpoint_name', model_name)
        results[model_name] = {"seen": {}, "unseen": {}}

        for env_id in seen_env_ids:
            results[model_name]["seen"][env_id] = {}
            for context_window in context_windows:
                print(f"\n=== Evaluating {model_name} on seen env {env_id} with context {context_window} ===")
                reasoning_result = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    env_id=env_id,
                    context_window=context_window,
                    num_envs=num_envs,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    reasoning_flag=True,
                    generation_kwargs=generation_kwargs,
                    device=device,
                    invalid_action_penalty=invalid_action_penalty,
                    consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                    model_name=model_name,
                    step_offset=current_step,
                )
                non_reasoning_result = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    env_id=env_id,
                    context_window=context_window,
                    num_envs=num_envs,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    reasoning_flag=False,
                    generation_kwargs=generation_kwargs,
                    device=device,
                    invalid_action_penalty=invalid_action_penalty,
                    consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                    model_name=model_name,
                    step_offset=current_step + num_batches,
                )
                results[model_name]["seen"][env_id][context_window] = {
                    "reasoning": reasoning_result["final_metrics"],
                    "non_reasoning": non_reasoning_result["final_metrics"],
                }
                current_step += 2 * num_batches

        for env_id in unseen_env_ids:
            results[model_name]["unseen"][env_id] = {}
            for context_window in context_windows:
                print(f"\n=== Evaluating {model_name} on unseen env {env_id} with context {context_window} ===")
                reasoning_result = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    env_id=env_id,
                    context_window=context_window,
                    num_envs=num_envs,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    reasoning_flag=True,
                    generation_kwargs=generation_kwargs,
                    device=device,
                    invalid_action_penalty=invalid_action_penalty,
                    consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                    model_name=model_name,
                    step_offset=current_step,
                )
                non_reasoning_result = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    env_id=env_id,
                    context_window=context_window,
                    num_envs=num_envs,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    reasoning_flag=False,
                    generation_kwargs=generation_kwargs,
                    device=device,
                    invalid_action_penalty=invalid_action_penalty,
                    consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                    model_name=model_name,
                    step_offset=current_step + num_batches,
                )
                results[model_name]["unseen"][env_id][context_window] = {
                    "reasoning": reasoning_result["final_metrics"],
                    "non_reasoning": non_reasoning_result["final_metrics"],
                }
                current_step += 2 * num_batches

        for context_window in context_windows:
            seen_baseline = {env_id: baseline_results["seen"][env_id][context_window] for env_id in seen_env_ids}
            seen_reasoning = {env_id: results[model_name]["seen"][env_id][context_window]["reasoning"]["success_rate"] for env_id in seen_env_ids}
            seen_non_reasoning = {env_id: results[model_name]["seen"][env_id][context_window]["non_reasoning"]["success_rate"] for env_id in seen_env_ids}

            plot_performance(
                env_ids=seen_env_ids,
                context_window=context_window,
                baseline_results=seen_baseline,
                reasoning_results=seen_reasoning,
                non_reasoning_results=seen_non_reasoning,
                env_type="Seen",
                checkpoint_name=checkpoint_name,
            )

            unseen_baseline = {env_id: baseline_results["unseen"][env_id][context_window] for env_id in unseen_env_ids}
            unseen_reasoning = {env_id: results[model_name]["unseen"][env_id][context_window]["reasoning"]["success_rate"] for env_id in unseen_env_ids}
            unseen_non_reasoning = {env_id: results[model_name]["unseen"][env_id][context_window]["non_reasoning"]["success_rate"] for env_id in unseen_env_ids}

            plot_performance(
                env_ids=unseen_env_ids,
                context_window=context_window,
                baseline_results=unseen_baseline,
                reasoning_results=unseen_reasoning,
                non_reasoning_results=unseen_non_reasoning,
                env_type="Unseen",
                checkpoint_name=checkpoint_name,
            )

    return results

# Load baseline model once
baseline_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
baseline_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    padding_side="left"
)

# Initialize models_info with the baseline model
models_info = [
    {
        'model': baseline_model,
        'tokenizer': baseline_tokenizer,
        'name': 'llama-3.2-3b-baseline'
    }
]

# Checkpoints
checkpoints = [
    ("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "0584ccf6786fc1733a7af991de032cf7dca00785"),  # Checkpoint 100
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "7c7fa7274f86f653a9db03aab81d2ab6e5a22d7f"),  # Checkpoint 90
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "7f8b9386d2b27dba1484255e39eb82241e902c62"),  # Checkpoint 80
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "33952a7e415f6463b972e6273dc3019c4bf7f489"),  # Checkpoint 70
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "9d7f1d2930977248ba6a2b6f006c917b4f06df2a"),  # Checkpoint 60
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "4275fb5ec80c9f84c03474aeedf0c81af8cf472e"),  # Checkpoint 50
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "0c899f429c008150d05eb2d94f9b2f1e40933ecb"),  # Checkpoint 40
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "d04ac0bcb78e7d803ad1d947e318b4e52b17080a"),  # Checkpoint 30
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "d287a43a11c1c3ebd9e04016b4fdec66791cc323"),  # Checkpoint 20
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "67728556b8cff0b3c18556a26e62315216c1dd44"),  # Checkpoint 10
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "0167cdea75d80598f4d674e14b4cb15f57a3ad96"),  # Initial commit
]

# Add each checkpoint as a trained model
for idx, (checkpoint, commit_hash) in enumerate(checkpoints):
    peft_config = PeftConfig.from_pretrained(checkpoint, revision=commit_hash)
    peft_model = PeftModel.from_pretrained(baseline_model, checkpoint, revision=commit_hash)
    checkpoint_short_name = f"Checkpoint_{(60 - idx*10)}"
    
    models_info.append({
        'model': peft_model,
        'tokenizer': baseline_tokenizer,  # Reuse baseline tokenizer
        'name': f'llama-3.2-3b-trained-{idx}',
        'checkpoint_name': checkpoint_short_name
    })

# Configure tokenizers and pad tokens
for model_info in models_info:
    model_info['model'].config.pad_token_id = model_info['tokenizer'].eos_token_id
    tokenizer = model_info['tokenizer']
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Setting pad_token to eos_token for {model_info['name']}")

seen_env_ids = ["BabyAI-GoTo-v0", "BabyAI-Pickup-v0"]
unseen_env_ids = ["BabyAI-Open-v0", "BabyAI-PutNext-v0", "BabyAI-PickUpSeqGoTo-v0"]

results = evaluate_models(
    models_info=models_info,
    seen_env_ids=seen_env_ids,
    unseen_env_ids=unseen_env_ids,
    context_windows=[5],
    num_batches=10,
)

summary_file = "outputs/logs/evaluation_summary.txt"
with open(summary_file, "w") as f:
    f.write("=== Evaluation Summary ===\n")
    for model_name, env_types in results.items():
        f.write(f"{model_name}:\n")
        for env_type, env_results in env_types.items():
            f.write(f"  {env_type}:\n")
            for env_id, context_results in env_results.items():
                for context_window, metrics in context_results.items():
                    f.write(f"    {model_name} on {env_id} ({env_type}) with context {context_window}:\n")
                    for metric_type, metric_values in metrics.items():
                        f.write(f"      {metric_type}:\n")
                        # Loop over each metric. If a corresponding _std is present,
                        # print the value plus/minus the standard deviation.
                        # We won't print the std separately.
                        for metric_name, value in metric_values.items():
                            std_key = f"{metric_name}_std"
                            if std_key in metric_values:
                                std_value = metric_values[std_key]
                                f.write(f"        {metric_name}: {value:.4f} ± {std_value:.4f}\n")
                            else:
                                f.write(f"        {metric_name}: {value:.4f}\n")
    f.write("\n")
print(f"Evaluation summary logged to {summary_file}")