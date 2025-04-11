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

        queries, responses, rewards, stats, running_stats = sample_batch(
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

        # Log query-output pairs
        decoded_queries = [tokenizer.decode(q, skip_special_tokens=True) if isinstance(q, torch.Tensor) else q for q in queries]
        decoded_responses = [tokenizer.decode(r, skip_special_tokens=True) if isinstance(r, torch.Tensor) else r for r in responses]

        query_response_pairs = []
        for query, response in zip(decoded_queries, decoded_responses):
            pair = f"Query: {query}\nOutput: {response}"
            query_response_pairs.append(pair)

        # Print to console
        print(f"Step {batch_idx + step_offset} - Model: {model_name}, Env: {env_id}")
        print("Query-Output Pairs:")
        for pair in query_response_pairs:
            print(pair)
            print("-" * 50)

        # Save to file
        with open(f"outputs/logs/model_outputs_{model_name}_{env_id}.txt", "a") as f:
            f.write(f"Step {batch_idx + step_offset} - Model: {model_name}, Env: {env_id}\n")
            f.write("Query-Output Pairs:\n")
            for pair in query_response_pairs:
                f.write(pair + "\n")
                f.write("-" * 50 + "\n")

    metrics = {
        "num_episodes": sum(num_episodes_per_batch),
        "success_rate": sum(successs) / len(successs) if successs else 0,
        "avg_reward": sum(rewardss) / len(rewardss) if rewardss else 0,
        "avg_steps": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
        "invalid_action_rate": sum(num_invalid_actions) / len(num_invalid_actions) if num_invalid_actions else 0,
        "avg_steps_to_success": (
            sum([length for length, success in zip(episode_lengths, successs) if success]) /
            sum(successs) if sum(successs) > 0 else float("nan")
        ),
        "avg_total_time": sum(total_times) / len(total_times),
        "avg_generate_time": sum(total_generate_times) / len(total_generate_times),
    }

    return {
        "final_metrics": metrics,
        "running_success_rates": running_success_rates,
        "running_avg_rewards": running_avg_rewards,
        "batch_times": batch_times,
    }

# Modified Plotting function to create a bar chart
def plot_performance(
    env_ids: List[str],  # List of environment IDs to plot
    context_window: int,
    baseline_results: Dict[str, float],  # Baseline success rates per env
    reasoning_results: Dict[str, float],  # Reasoning success rates per env
    non_reasoning_results: Dict[str, float],  # Non-reasoning success rates per env
    env_type: str,
):
    # Set up the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Number of environments
    n_envs = len(env_ids)
    bar_width = 0.25  # Width of each bar
    index = np.arange(n_envs)  # X-axis positions for each environment

    # Plot bars for each model
    ax.bar(index - bar_width, [baseline_results[env_id] for env_id in env_ids], 
           bar_width, label="Baseline (Untrained)", color='red', alpha=0.7)
    ax.bar(index, [reasoning_results[env_id] for env_id in env_ids], 
           bar_width, label="Reasoning (Trained)", color='blue', alpha=0.7)
    ax.bar(index + bar_width, [non_reasoning_results[env_id] for env_id in env_ids], 
           bar_width, label="Non-Reasoning (Trained)", color='green', alpha=0.7)

    # Customize the plot
    ax.set_xlabel("Environments", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(f"Model Performance Comparison (Context Window: {context_window}, {env_type})", fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(env_ids, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot
    plot_path = f"outputs/plots/{env_type.lower()}_comparison_ctx{context_window}.png"
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

    # Find the baseline (untrained) model
    baseline_model_info = None
    for model_info in models_info:
        if "untrained" in model_info['name'].lower():
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

    # Evaluate the trained model (reasoning and non-reasoning)
    for model_info in models_info:
        if "untrained" in model_info['name'].lower():
            continue

        model = model_info['model']
        tokenizer = model_info['tokenizer']
        model_name = model_info['name']
        results[model_name] = {"seen": {}, "unseen": {}}

        # Evaluate on seen environments
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

        # Evaluate on unseen environments
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

    # After evaluating all models, generate bar charts for each context window
    for context_window in context_windows:
        # Seen environments
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
        )

        # Unseen environments
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
        )

    return results

# Example usage
checkpoint = "pavanpreet-gandhi/babyai-ppo-2025-03-30_11-36-26"
peft_config = PeftConfig.from_pretrained(checkpoint)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
peft_model = PeftModel.from_pretrained(base_model, checkpoint)

models_info = [
    {
        'model': AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ),
        'tokenizer': AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            padding_side="left"
        ),
        'name': 'llama-3.2-3b-untrained'
    },
    {
        'model': peft_model,
        'tokenizer': AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            padding_side="left"
        ),
        'name': 'llama-3.2-3b-trained'
    }
]

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
    context_windows=[1, 3, 5],
    num_batches=5,
)

print("\n=== Evaluation Summary ===")
for model_name, env_types in results.items():
    for env_type, env_results in env_types.items():
        for env_id, context_results in env_results.items():
            for context_window, metrics in context_results.items():
                print(f"{model_name} on {env_id} ({env_type}) with context {context_window}:")
                for metric_type, metric_values in metrics.items():
                    print(f"  {metric_type}:")
                    for metric_name, value in metric_values.items():
                        print(f"    {metric_name}: {value:.4f}")