# Import necessary libraries
import numpy as np
import torch
import wandb
import gym
import time
import matplotlib.pyplot as plt
from env_manager import EnvManager
import babyai_text
import utils
from peft import PeftModel, PeftConfig
from typing import Dict, Any, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from src import sample_batch
babyai_text.register_levels(__name__, globals())

# Evaluation function to evaluate a model on a specific environment
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
    log_to_wandb: bool = True,
    model_name: str = "model",
    step_offset: int = 0,
) -> Dict[str, float]:
    """
    Evaluates a model on a specific environment with batch-based sampling, matching baseline config.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate;
        tokenizer: Tokenizer for processing input;
        env_id (str): The environment ID;
        context_window (int): Size of the context window;
        num_envs (int): Number of parallel environments;
        num_batches (int): Number of batches to evaluate;
        batch_size (int): Batch size for sampling;
        reasoning_flag (bool): Whether to use reasoning ;
        generation_kwargs (dict, optional): Dictionary of generation parameters;
        device (torch.device, optional): Device to run on;
        invalid_action_penalty (float): Penalty for invalid actions;
        consecutive_invalid_actions_allowed (int): Max consecutive invalid actions;
        log_to_wandb (bool): Whether to log to Weights & Biases;
        model_name (str): Name of the model for logging;
        step_offset (int): Offset for step numbering in wandb.

    Returns:
        Dict[str, float]: Dictionary with evaluation metrics.
    """
    print(f"Evaluating {env_id} with {num_batches} batches...")

    # Set default generation_kwargs if not provided
    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.95,
            "temperature": 0.8,
        }

    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize parallel environments
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

    # Lists to collect metrics across batches
    total_times = []
    total_generate_times = []
    num_episodes_per_batch = []
    successs = []
    rewardss = []
    episode_lengths = []
    num_invalid_actions = []

    # Run batches
    for batch_idx in range(num_batches):
        start_time = time.time()
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
        end_time = time.time()
        batch_time = end_time - start_time

        # Collect batch-level metrics
        total_times.append(batch_time)
        total_generate_times.append(stats["total_generate_time"])
        num_episodes = len(running_stats['success'][env_id])
        num_episodes_per_batch.append(num_episodes)

        # Aggregate per-episode metrics
        batch_success = running_stats['success'][env_id]
        batch_rewards = running_stats['rewards'][env_id]
        batch_lengths = running_stats['episode_lengths'][env_id]
        batch_invalids = running_stats['num_invalid_actions'][env_id]

        successs.extend(batch_success)
        rewardss.extend(batch_rewards)
        episode_lengths.extend(batch_lengths)
        num_invalid_actions.extend(batch_invalids)

        # Log per-batch metrics to wandb
        if log_to_wandb:
            batch_step = step_offset + batch_idx
            wandb.log({
                f"{model_name}/{env_id}/batch/success_rate": sum(batch_success) / len(batch_success) if batch_success else 0,
                f"{model_name}/{env_id}/batch/avg_reward": sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0,
                f"{model_name}/{env_id}/batch/avg_length": sum(batch_lengths) / len(batch_lengths) if batch_lengths else 0,
                f"{model_name}/{env_id}/batch/avg_invalid": sum(batch_invalids) / len(batch_invalids) if batch_invalids else 0,
                f"{model_name}/{env_id}/batch/time": batch_time,
                f"{model_name}/{env_id}/batch/generate_time": stats["total_generate_time"],
                f"{model_name}/{env_id}/running/success_rate": sum(successs) / len(successs) if successs else 0,
                f"{model_name}/{env_id}/running/avg_reward": sum(rewardss) / len(rewardss) if rewardss else 0,
                f"{model_name}/{env_id}/running/avg_invalid": sum(num_invalid_actions) / len(num_invalid_actions) if num_invalid_actions else 0,
            }, step=batch_step)

    # Compute summary metrics
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
    return metrics

# Function to evaluate multiple models on multiple environments
def evaluate_models(
    models_info: List[Dict[str, Any]],
    env_ids: List[str],
    context_windows: List[int] = [1, 2, 3, 4, 5],
    num_envs: int = 4,
    num_batches: int = 3,
    batch_size: int = 128,
    reasoning_flag: bool = False,
    generation_kwargs: dict = None,
    device: torch.device = None,
    invalid_action_penalty: float = -2,
    consecutive_invalid_actions_allowed: int = 5,
    log_to_wandb: bool = True,
    step_offset: int = 0,
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """
    Evaluates multiple models on multiple environments with varying context windows.

    Args:
        models_info (List[Dict[str, Any]]): List of dicts with 'model', 'tokenizer', and 'name';
        env_ids (List[str]): List of environment IDsl;
        context_windows (List[int]): List of context window sizes;
        num_envs (int): Number of parallel environments;
        num_batches (int): Number of batches to run;
        batch_size (int): Batch size for sampling;
        reasoning_flag (bool): Whether to use reasoning;
        generation_kwargs (dict, optional): Generation parameters;
        device (torch.device, optional): Device to run on;
        invalid_action_penalty (float): Penalty for invalid actions;
        consecutive_invalid_actions_allowed (int): Max consecutive invalid actions;
        log_to_wandb (bool): Whether to log to Weights & Biases;
        step_offset (int): Starting step offset for wandb logging.

    Returns:
        Dict[str, Dict[str, Dict[int, Dict[str, float]]]]: Nested results {model: {env: {context: metrics}}}.
    """
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

    for model_info in models_info:
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        model_name = model_info['name']
        results[model_name] = {}

        for env_id in env_ids:
            results[model_name][env_id] = {}
            for context_window in context_windows:
                print(f"\n=== Evaluating model: {model_name} on {env_id} with context window: {context_window} ===")
                metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    env_id=env_id,
                    context_window=context_window,
                    num_envs=num_envs,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    reasoning_flag=reasoning_flag,
                    generation_kwargs=generation_kwargs,
                    device=device,
                    invalid_action_penalty=invalid_action_penalty,
                    consecutive_invalid_actions_allowed=consecutive_invalid_actions_allowed,
                    log_to_wandb=log_to_wandb,
                    model_name=model_name,
                    step_offset=current_step,
                )
                results[model_name][env_id][context_window] = metrics

                # Log summary metrics to wandb
                if log_to_wandb:
                    for metric_name, value in metrics.items():
                        wandb.log({
                            f"eval/{model_name}/{env_id}/ctx{context_window}/{metric_name}": value
                        }, step=current_step + num_batches)

                # Print results
                for k, v in metrics.items():
                    print(f"    {k}: {v:.4f}")

                # Increment step offset for the next evaluation
                current_step += num_batches
    
    return results

# Example usage

# Initialize wandb project and name for logging
wandb.init(project="Evaluation", name="llama-32-3b-eval")

# Define the models you want to evaluate, can be multiple models
models_info = [{
    'model': AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ),
    'tokenizer': AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", 
        padding_side="left"
    ),
    'name': 'llama-3.2-3b'
}]

# Define the environments you want to evaluate on
env_ids = ["BabyAI-GoTo-v0", "BabyAI-Pickup-v0", "BabyAI-Open-v0", "BabyAI-PutNext-v0", "BabyAI-PickUpSeqGoTo-v0"]


# Set the padding token
for model_info in models_info:
    tokenizer = model_info['tokenizer']
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Setting pad_token to eos_token for {model_info['name']}")

# Adjust parameters if needed HERE
results = evaluate_models(
    models_info=models_info,
    env_ids=env_ids,
    context_windows=[3],
    num_batches = 3
)

# Print summary
print("\n=== Evaluation Summary ===")
for model_name, env_results in results.items():
    for env_id, context_results in env_results.items():
        for context_window, metrics in context_results.items():
            print(f"{model_name} on {env_id} with context {context_window}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

wandb.finish()