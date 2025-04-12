import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from src import EnvManager
from sample_episodes import sample_episodes
import babyai_text
from peft import PeftModel, PeftConfig
from typing import Dict, Any, List, Union
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import logging
import os


# Set PyTorch CUDA allocator config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress transformers logging to ERROR level
logging.getLogger("transformers").setLevel(logging.ERROR)

# Create directories for outputs
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)


babyai_text.register_levels(__name__, globals())

def evaluate(
    model,
    tokenizer,
    env_id: str,
    context_window: int,
    num_envs: int = 1,
    num_episodes: int = 50,
    reasoning_flag: bool = False,
    generation_kwargs: dict = None,
    device: torch.device = None,
    invalid_action_penalty: float = -0.1,
    consecutive_invalid_actions_allowed: int = 5,
    model_name: str = "model",
    step_offset: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate a model on BabyAI tasks by running a fixed number of episodes.
    This version directly uses sample_episodes(...) to gather rollouts.
    """

    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.8,
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating {model_name} on {env_id} with {num_episodes} episodes (reasoning={reasoning_flag})...")

    # Move model to device and set evaluation mode
    model.to(device)
    model.eval()

    # If the model does not have a generation_config, inject one.
    if not hasattr(model, "generation_config"):
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig(**generation_kwargs)

    # Create multiple EnvManager instances
    envs = [
        EnvManager(
            env_ids=[env_id],
            invalid_action_penalty=invalid_action_penalty,
            consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
            reasoning_flag=reasoning_flag
        )
        for _ in range(num_envs)
    ]

    # Run sample_episodes to gather rollouts (assuming sample_episodes is imported)
    eval_start_time = time.time()
    stats, contexts = sample_episodes(
        envs=envs,
        tokenizer=tokenizer,
        model=model,
        generation_kwargs=generation_kwargs,
        device=device,
        number_of_episodes=num_episodes,
        context_window=context_window,
        reasoning_flag=reasoning_flag,
    )
    eval_end_time = time.time()
    

    # Basic metrics from stats
    # stats includes: "success", "rewards", "episode_lengths", "num_invalid_actions"
    num_samples = len(stats["success"])
    if num_samples == 0:
        # Edge case: if no episodes were actually run
        return {
            "final_metrics": {
                "num_episodes": 0,
                "success_rate": 0.0,
                "success_rate_std": 0.0,
                "avg_reward": 0.0,
                "avg_reward_std": 0.0,
                "avg_steps": 0.0,
                "avg_steps_std": 0.0,
                "invalid_action_rate": 0.0,
                "invalid_action_rate_std": 0.0,
                "avg_steps_to_success": float("nan"),
                "avg_steps_to_success_std": float("nan"),
                "avg_total_time": 0.0,
                "avg_generate_time": 0.0,
            },
            "running_success_rates": [],
            "running_avg_rewards": [],
            "episode_times": [],
        }

    # Convert stats to tensors for easy math
    success_tensor = torch.tensor(stats["success"], dtype=torch.float32)
    reward_tensor = torch.tensor(stats["rewards"], dtype=torch.float32)
    length_tensor = torch.tensor(stats["episode_lengths"], dtype=torch.float32)
    invalid_tensor = torch.tensor(stats["num_invalid_actions"], dtype=torch.float32)

    success_rate = success_tensor.mean().item()
    success_rate_std = success_tensor.std().item() if num_samples > 1 else 0.0

    avg_reward = reward_tensor.mean().item()
    avg_reward_std = reward_tensor.std().item() if num_samples > 1 else 0.0

    avg_steps = length_tensor.mean().item()
    avg_steps_std = length_tensor.std().item() if num_samples > 1 else 0.0

    invalid_rate = invalid_tensor.mean().item()
    invalid_rate_std = invalid_tensor.std().item() if num_samples > 1 else 0.0

    # Steps to success (for episodes that succeeded)
    successful_indices = (success_tensor == 1)
    if successful_indices.any():
        steps_for_successes = length_tensor[successful_indices]
        avg_steps_success = steps_for_successes.mean().item()
        avg_steps_success_std = steps_for_successes.std().item() if len(steps_for_successes) > 1 else 0.0
    else:
        avg_steps_success = float("nan")
        avg_steps_success_std = float("nan")

    # We didn't track generation time for each step in sample_episodes(...) by default
    # If you'd like, you can modify sample_episodes to accumulate generation_time and return it in stats.

    total_time = eval_end_time - eval_start_time

    # Return final dictionary consistent with your typical format
    metrics = {
        "num_episodes": num_samples,
        "success_rate": success_rate,
        "success_rate_std": success_rate_std,
        "avg_reward": avg_reward,
        "avg_reward_std": avg_reward_std,
        "avg_steps": avg_steps,
        "avg_steps_std": avg_steps_std,
        "invalid_action_rate": invalid_rate,
        "invalid_action_rate_std": invalid_rate_std,
        "avg_steps_to_success": avg_steps_success,
        "avg_steps_to_success_std": avg_steps_success_std,
        "avg_total_time": total_time,          # total for the entire run
        "avg_total_time_std": 0.0,            # we only did one entire run
        "avg_generate_time": float("nan"),    # can’t be computed unless sample_episodes returns timing
        "avg_generate_time_std": float("nan"),
    }

    # If you want “running_success_rates,” “running_avg_rewards,” or “episode_times,”
    # you can easily track them in sample_episodes(...) as well.
    # For now, we’ll return empty lists for them by default.
    return {
        "final_metrics": metrics,
        "running_success_rates": [],
        "running_avg_rewards": [],
        "episode_times": [],
    }
# Plotting function (unchanged)
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

# Evaluation function on multiple models (minor adjustments)
def evaluate_models(
    models_info: List[Dict[str, Any]],
    seen_env_ids: List[str],
    unseen_env_ids: List[str],
    context_windows: List[int] = [5],
    num_envs: int = 6,  # Aligned with notebook
    num_episodes: int = 50,
    generation_kwargs: dict = None,
    device: torch.device = None,
    invalid_action_penalty: float = -2.0,  # Aligned with notebook
    consecutive_invalid_actions_allowed: int = 5,
    step_offset: int = 0,
) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "top_k": 50,
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

    # Evaluate baseline model
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
                num_episodes=num_episodes,
                reasoning_flag=False,
                generation_kwargs=generation_kwargs,
                device=device,
                invalid_action_penalty=invalid_action_penalty,
                consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                model_name=baseline_model_info['name'],
                step_offset=current_step,
            )
            baseline_results["seen"][env_id][context_window] = result["final_metrics"]["success_rate"]
            current_step += num_episodes

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
                num_episodes=num_episodes,
                reasoning_flag=False,
                generation_kwargs=generation_kwargs,
                device=device,
                invalid_action_penalty=invalid_action_penalty,
                consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                model_name=baseline_model_info['name'],
                step_offset=current_step,
            )
            baseline_results["unseen"][env_id][context_window] = result["final_metrics"]["success_rate"]
            current_step += num_episodes

    # Evaluate trained models
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
                    num_episodes=num_episodes,
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
                    num_episodes=num_episodes,
                    reasoning_flag=False,
                    generation_kwargs=generation_kwargs,
                    device=device,
                    invalid_action_penalty=invalid_action_penalty,
                    consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                    model_name=model_name,
                    step_offset=current_step + num_episodes,
                )
                results[model_name]["seen"][env_id][context_window] = {
                    "reasoning": reasoning_result["final_metrics"],
                    "non_reasoning": non_reasoning_result["final_metrics"],
                }
                current_step += 2 * num_episodes

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
                    num_episodes=num_episodes,
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
                    num_episodes=num_episodes,
                    reasoning_flag=False,
                    generation_kwargs=generation_kwargs,
                    device=device,
                    invalid_action_penalty=invalid_action_penalty,
                    consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                    model_name=model_name,
                    step_offset=current_step + num_episodes,
                )
                results[model_name]["unseen"][env_id][context_window] = {
                    "reasoning": reasoning_result["final_metrics"],
                    "non_reasoning": non_reasoning_result["final_metrics"],
                }
                current_step += 2 * num_episodes

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

# Load baseline model
baseline_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
baseline_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    padding_side="left"
)

# Initialize models_info with baseline model
models_info = [
    {
        'model': baseline_model,
        'tokenizer': baseline_tokenizer,
        'name': 'llama-3.2-3b-baseline'
    }
]

# Checkpoints
checkpoints = [
    ("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "3a698a6adce4838068348f97e87dafddbf05be2d"),  # Checkpoint 160
    ("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "4b4784ff36b04aa34cc3f84bf154c30e133c9a36"),  # Checkpoint 150
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "7c27ab0a7fbd204138b01ca6b38a20adfda0128e"),  # Checkpoint 140
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "169b7890e98134d71d14707751e55f7eee54507a"),  # Checkpoint 130
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "0eb2d6751c2ba48a0d95302a10ee84f1b4ab274b"),  # Checkpoint 120
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "57c165bcb6db38a78b7d67846fbb2418041fca99"),  # Checkpoint 110
    #("pavanpreet-gandhi/babyai-classical-ppo-prefinal-experiments-2025-04-11_13-38-03", "0584ccf6786fc1733a7af991de032cf7dca00785"),  # Checkpoint 100
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
    checkpoint_short_name = f"Checkpoint_{(160 - idx*10)}"
    
    models_info.append({
        'model': peft_model,
        'tokenizer': baseline_tokenizer,
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

# Run evaluation
results = evaluate_models(
    models_info=models_info,
    seen_env_ids=seen_env_ids,
    unseen_env_ids=unseen_env_ids,
    context_windows=[5],
    num_episodes=50,
)

# Summary logging
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
                        for metric_name, value in metric_values.items():
                            std_key = f"{metric_name}_std"
                            if std_key in metric_values:
                                std_value = metric_values[std_key]
                                f.write(f"        {metric_name}: {value:.4f} ± {std_value:.4f}\n")
                            else:
                                f.write(f"        {metric_name}: {value:.4f}\n")
    f.write("\n")
print(f"Evaluation summary logged to {summary_file}")