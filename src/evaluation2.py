import numpy as np
import torch
from env_manager import EnvManager
import gym
import babyai_text
babyai_text.register_levels(__name__, globals())
import utils
from typing import Dict, Any, List
from torch.nn.utils.rnn import pad_sequence
import wandb
import matplotlib.pyplot as plt

def evaluate(
    model,
    tokenizer,
    generation_kwargs: Dict[str, Any],
    num_episodes: int,
    env_id: str,
    env_kwargs: Dict[str, Any] = {},
    num_envs: int = 4,
    context_window: int = 5,
    seed_offset: int = 1000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, float]:
    print(f"Evaluating {env_id} with {num_episodes} episodes...")
    try:
        eval_env_managers = [
            EnvManager(gym.make(env_id, seed=i + seed_offset, **env_kwargs))
            for i in range(num_envs)
        ]
        print("Evaluation environments initialized.")
    except Exception as e:
        print(f"Failed to initialize environments for {env_id}: {e}")
        raise

    system_prompt_template = utils.get_system_prompt()
    contexts: List[List[Dict[str, str]]] = [[] for _ in range(num_envs)]
    missions, text_obss = zip(*[env.reset() for env in eval_env_managers])
    for i, (context, mission, text_obs) in enumerate(zip(contexts, missions, text_obss)):
        system_prompt = system_prompt_template.replace("{goal}", mission)
        context.append({"role": "system", "content": system_prompt})
        context.append({"role": "user", "content": text_obs})
    print("Contexts initialized.")

    episode_stats = []
    current_episode_reward = [0.0] * num_envs
    current_episode_steps = [0] * num_envs
    current_episode_invalid_actions = [0] * num_envs
    valid_actions = set(utils.text_to_action.keys())

    while len(episode_stats) < num_episodes:
        print(f"Episode {len(episode_stats)+1}/{num_episodes}...")
        query_tensors_list = [
            tokenizer.apply_chat_template(context, return_tensors="pt", add_generation_prompt=True).squeeze(0)
            for context in contexts
        ]
        encoded = tokenizer.pad({"input_ids": query_tensors_list}, padding=True, return_tensors="pt").to(device)
        attention_mask = encoded["attention_mask"]

        response_tensors_step = model.generate(
            encoded["input_ids"],
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        generated_ids = [resp[len(enc):] for resp, enc in zip(response_tensors_step, encoded["input_ids"])]
        response_texts_step = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated_ids]

        actions = []
        for response_text in response_texts_step:
            action = next((act for act in valid_actions if act in response_text), "done")
            actions.append(action)

        print("\n---- MODEL OUTPUTS ----")
        print(f"Valid actions: {list(valid_actions)}")
        for i, (resp, act) in enumerate(zip(response_texts_step, actions)):
            print(f"Raw output {i}: '{resp}' -> Action: '{act}'")

        for i, (env, action) in enumerate(zip(eval_env_managers, actions)):
            text_obs, reward, done = env.step(action)
            current_episode_reward[i] += reward
            current_episode_steps[i] += 1
            if action not in valid_actions:
                current_episode_invalid_actions[i] += 1

            # Be more aggressive with context trimming
            max_messages = min(2 * context_window, 10) 

            contexts[i].append({"role": "assistant", "content": action})
            contexts[i].append({"role": "user", "content": text_obs})

            # Truncate older messages if the conversation grows too big
            if len(contexts[i]) > max_messages:
                contexts[i] = contexts[i][-max_messages:]

            if done:
                print(f"Episode finished in env {i}: steps={current_episode_steps[i]}, reward={current_episode_reward[i]}")
                success = 1 if reward > 0 else 0
                episode_stats.append({
                    "success": success,
                    "total_reward": current_episode_reward[i],
                    "steps": current_episode_steps[i],
                    "invalid_actions": current_episode_invalid_actions[i],
                })
                current_episode_reward[i] = 0.0
                current_episode_steps[i] = 0
                current_episode_invalid_actions[i] = 0
                mission, text_obs = env.reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_obs}
                ]

    success_rate = np.mean([stat["success"] for stat in episode_stats])
    avg_reward = np.mean([stat["total_reward"] for stat in episode_stats])
    avg_steps = np.mean([stat["steps"] for stat in episode_stats])
    total_invalid_actions = sum(stat["invalid_actions"] for stat in episode_stats)
    total_actions = sum(stat["steps"] for stat in episode_stats)
    invalid_action_rate = total_invalid_actions / total_actions if total_actions > 0 else 0.0
    successful_episodes = [stat for stat in episode_stats if stat["success"] == 1]
    avg_steps_to_success = (
        np.mean([stat["steps"] for stat in successful_episodes])
        if successful_episodes else float("nan")
    )

    metrics = {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "invalid_action_rate": invalid_action_rate,
        "avg_steps_to_success": avg_steps_to_success,
    }
    return metrics

# Example usage: Run evaluation on Lama-3b on goto & pickup tasks
if __name__ == "__main__":
    print("Script started")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import AutoModelForCausalLMWithValueHead
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("Loading tokenizer...")
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 
        print("Tokenizer loaded.")

        print("Loading model...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(device)
        model.eval()
        print("Model loaded.")

        generation_kwargs = {
            "max_new_tokens": 3,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        task_types = ["BabyAI-GoToObj-v0", "BabyAI-Pickup-v0"]
        num_eval_episodes = 10
        num_envs = 4
        context_window = 5
        seed_offset = 1000

        # Initialize W&B with fallback for offline mode
        print("Initializing Weights & Biases...")
        try:
            wandb.init(project="babyai_evaluation", name="llama-3b-evaluation")
        except Exception as e:
            print(f"W&B initialization failed: {e}. Running in offline mode.")
            wandb.init(project="babyai_evaluation", name="llama-3b-evaluation", mode="offline")

        all_metrics = {}
        for env_id in task_types:
            print(f"\nStarting evaluation for {env_id}...")
            try:
                metrics = evaluate(
                    model,
                    tokenizer,
                    generation_kwargs,
                    num_eval_episodes,
                    env_id,
                    num_envs=num_envs,
                    context_window=context_window,
                    seed_offset=seed_offset,
                    device=device,
                )
                print(f"Evaluation completed for {env_id}.")
                print(f"Metrics for {env_id}:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value}")
                wandb.log({f"{env_id}/{k}": v for k, v in metrics.items()})
                all_metrics[env_id] = metrics
            except Exception as e:
                print(f"Evaluation failed for {env_id}: {e}")

        metrics_to_plot = ["success_rate", "avg_reward", "avg_steps", "invalid_action_rate", "avg_steps_to_success"]
        task_names = list(all_metrics.keys())

        # Plotting on wandb
        if all_metrics:
            for metric in metrics_to_plot:
                try:
                    plt.figure(figsize=(10, 6))
                    # Filter out NaN values
                    values = []
                    labels = []
                    for task in task_names:
                        if metric in all_metrics[task] and not (isinstance(all_metrics[task][metric], float) and np.isnan(all_metrics[task][metric])):
                            values.append(all_metrics[task][metric])
                            labels.append(task)
                    
                    if not values:  # Skip if no valid values
                        print(f"No valid data for {metric}, skipping plot")
                        continue
                        
                    plt.bar(labels, values, color='skyblue')
                    plt.title(f"{metric.replace('_', ' ').title()} Across Tasks")
                    plt.xlabel("Task Type")
                    plt.ylabel(metric.replace('_', ' ').title())
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    wandb.log({f"plot/{metric}": wandb.Image(plt)})
                    plt.close()
                except Exception as e:
                    print(f"Failed to plot {metric}: {e}")
        
        wandb.finish()  # Close wandb run
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        if wandb.run is not None:
            wandb.finish()