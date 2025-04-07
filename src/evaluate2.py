import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

# Example placeholders for your environment and utility modules
from env_manager import EnvManager
import babyai_text
import utils
babyai_text.register_levels(__name__, globals())

from typing import Dict, Any, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

###############################################################################
# EVALUATION FUNCTION
###############################################################################

def evaluate(
    model: torch.nn.Module,
    tokenizer,
    generation_kwargs: Dict[str, Any],
    num_episodes: int,
    env_id: str,
    env_kwargs: Dict[str, Any] = {},
    num_envs: int = 4,
    context_window: int = 5,
    seed_offset: int = 1000,
    device: Union[str, torch.device] = "cuda",
) -> Dict[str, float]:
    """
    Evaluate a model on a specific environment for a certain number of episodes.
    Returns a dictionary of metrics: success_rate, avg_reward, avg_steps, etc.
    """
    print(f"Evaluating {env_id} with {num_episodes} episodes...")

    # Initialize parallel environments
    try:
        eval_env_managers = [
            EnvManager(gym.make(env_id, seed=i + seed_offset, **env_kwargs))
            for i in range(num_envs)
        ]
    except Exception as e:
        print(f"Failed to initialize environments for {env_id}: {e}")
        raise

    system_prompt_template = utils.get_system_prompt()
    contexts: List[List[Dict[str, str]]] = [[] for _ in range(num_envs)]
    missions, text_obss = zip(*[env.reset() for env in eval_env_managers])

    # Initialize conversation contexts
    for i, (context, mission, text_obs) in enumerate(zip(contexts, missions, text_obss)):
        system_prompt = system_prompt_template.replace("{goal}", mission)
        context.append({"role": "system", "content": system_prompt})
        context.append({"role": "user", "content": text_obs})

    episode_stats = []
    current_episode_reward = [0.0] * num_envs
    current_episode_steps = [0] * num_envs
    current_episode_invalid_actions = [0] * num_envs
    valid_actions = set(utils.text_to_action.keys())

    # Step until we gather num_episodes total across all envs
    while len(episode_stats) < num_episodes:
        query_tensors_list = []
        for context in contexts:
            # Convert context -> token ids for the model
            input_ids = tokenizer.apply_chat_template(
                context, return_tensors="pt", add_generation_prompt=True
            ).squeeze(0)
            query_tensors_list.append(input_ids)

        # Pad all queries in this step
        encoded = tokenizer.pad({"input_ids": query_tensors_list}, 
                                padding=True, return_tensors="pt").to(device)
        attention_mask = encoded["attention_mask"]

        # Generate step outputs
        response_tensors_step = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        # For each environment, decode only the newly generated tokens
        generated_ids = [
            resp[len(enc):]  # slice off the prompt portion
            for resp, enc in zip(response_tensors_step, encoded["input_ids"])
        ]
        response_texts_step = [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]

        # Map raw text outputs -> single action from valid actions, defaulting to "done"
        actions = []
        for response_text in response_texts_step:
            # A naive approach: pick first valid action found in the text
            action = next((act for act in valid_actions if act in response_text), "done")
            actions.append(action)

        # Step in each environment with the chosen action
        for i, (env, action) in enumerate(zip(eval_env_managers, actions)):
            text_obs, reward, done = env.step(action)
            current_episode_reward[i] += reward
            current_episode_steps[i] += 1
            if action not in valid_actions:
                current_episode_invalid_actions[i] += 1

            # Append the model's action + next user observation to context
            contexts[i].append({"role": "assistant", "content": action})
            contexts[i].append({"role": "user", "content": text_obs})

            # Aggressive context trimming
            max_messages = min(2 * context_window, 10)
            if len(contexts[i]) > max_messages:
                contexts[i] = contexts[i][-max_messages:]

            # If environment is done, log episode stats and reset
            if done:
                success = 1 if reward > 0 else 0
                episode_stats.append({
                    "success": success,
                    "total_reward": current_episode_reward[i],
                    "steps": current_episode_steps[i],
                    "invalid_actions": current_episode_invalid_actions[i],
                })

                # Reset counters
                current_episode_reward[i] = 0.0
                current_episode_steps[i] = 0
                current_episode_invalid_actions[i] = 0

                # Reset environment + context
                mission, text_obs = env.reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_obs},
                ]

    # Compute summary metrics
    success_rate = np.mean([stat["success"] for stat in episode_stats])
    avg_reward = np.mean([stat["total_reward"] for stat in episode_stats])
    avg_steps = np.mean([stat["steps"] for stat in episode_stats])
    total_invalid_actions = sum(stat["invalid_actions"] for stat in episode_stats)
    total_actions = sum(stat["steps"] for stat in episode_stats)
    invalid_action_rate = total_invalid_actions / total_actions if total_actions > 0 else 0.0
    successful_episodes = [s for s in episode_stats if s["success"] == 1]
    avg_steps_to_success = (
        np.mean([s["steps"] for s in successful_episodes]) if successful_episodes else float("nan")
    )

    metrics = {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "invalid_action_rate": invalid_action_rate,
        "avg_steps_to_success": avg_steps_to_success,
    }
    return metrics

###############################################################################
# MAIN SCRIPT EXAMPLE:
###############################################################################

if __name__ == "__main__":
    print("Script started")

    # You can extend this list with local paths to fine-tuned checkpoints or other models
    model_list = [
        "meta-llama/Llama-3.2-3B-Instruct",  # baseline
        # Add more local or remote checkpoints here, e.g.:
        # "/path/to/finetuned-checkpoint-1000",
        # "/path/to/finetuned-checkpoint-2000",
        # ...
    ]

    task_types = ["BabyAI-GoToObj-v0", "BabyAI-Pickup-v0"]
    num_eval_episodes = 10
    num_envs = 4
    context_window = 5
    seed_offset = 1000

    # Generation settings for each inference call
    generation_kwargs = {
        "max_new_tokens": 3,
        "do_sample": False,
        "repetition_penalty": 1.0,
        # The following might be needed depending on your model tokenizer
        # "pad_token_id": tokenizer.eos_token_id,
        # "eos_token_id": tokenizer.eos_token_id,
    }

    # We'll store evaluation results in a nested dict:
    # results[model_name][env_id] = { "success_rate": ..., "avg_reward": ..., etc. }
    all_results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model_id in model_list:
        print(f"\n=== Evaluating model: {model_id} ===")
        # Load tokenizer & model
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()

        model_results = {}
        for env_id in task_types:
            print(f"--> Evaluating on task: {env_id}")
            try:
                metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    generation_kwargs=generation_kwargs,
                    num_episodes=num_eval_episodes,
                    env_id=env_id,
                    num_envs=num_envs,
                    context_window=context_window,
                    seed_offset=seed_offset,
                    device=device
                )
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                model_results[env_id] = metrics
            except Exception as e:
                print(f"Evaluation failed for {env_id} with error: {e}")
                model_results[env_id] = None

        all_results[model_id] = model_results

    # -------------------------------------------------------------------------
    # PLOTTING RESULTS
    # -------------------------------------------------------------------------
    # We'll plot each metric in a separate bar chart for all models & tasks.

    # Which metrics do we want to plot?
    metrics_to_plot = [
        "success_rate",
        "avg_reward",
        "avg_steps",
        "invalid_action_rate",
        "avg_steps_to_success",
    ]

    # If you have multiple tasks, you can choose to plot them side-by-side or
    # create separate figures for each task. Here, for simplicity, we'll create
    # a separate figure *per metric* and show bars for each (model, task).

    # Make sure we do not create subplots in a single figure.
    # We'll do one figure per metric, each containing multiple bars.

    for metric in metrics_to_plot:
        # Prepare data
        x_labels = []
        values = []
        for model_id, tasks_dict in all_results.items():
            # For each model, we might average the metric across tasks or keep them separate
            # If you want separate bars per (model, task), you can break it down further.
            for env_id, env_metrics in tasks_dict.items():
                if env_metrics is None:
                    continue
                val = env_metrics.get(metric, float("nan"))
                label = f"{model_id} - {env_id}"
                x_labels.append(label)
                values.append(val)

        # If we have no valid data, skip plotting
        if not values:
            print(f"No valid data for metric '{metric}'. Skipping.")
            continue

        plt.figure()  # new figure for each metric
        plt.bar(x_labels, values)
        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Model / Environment")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save or display the figure as needed. Example: save to disk
        plt.savefig(f"plot_{metric}.png", dpi=150)
        plt.close()

    print("Evaluation complete. Plots saved.")