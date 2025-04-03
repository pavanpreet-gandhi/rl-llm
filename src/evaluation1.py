import numpy as np
import torch
from env_manager import EnvManager
import gym
import babyai_text
babyai_text.register_levels(__name__, globals())
import utils
from typing import Dict, Any, List
from torch.nn.utils.rnn import pad_sequence

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
    """
    Evaluate a language model in the BabyAI environment over a specified number of episodes.
    """
    # Initialize evaluation environments with custom configurations
    print("Initializing evaluation environments...")
    eval_env_managers = [
        EnvManager(gym.make(env_id, seed=i + seed_offset, **env_kwargs))
        for i in range(num_envs)
    ]

    # Set up system prompt template
    system_prompt_template = utils.get_system_prompt()

    # Initialize contexts for each environment
    contexts: List[List[Dict[str, str]]] = [[] for _ in range(num_envs)]
    missions, text_obss = zip(*[env.reset() for env in eval_env_managers])
    for i, (context, mission, text_obs) in enumerate(zip(contexts, missions, text_obss)):
        system_prompt = system_prompt_template.replace("{goal}", mission)
        context.append({"role": "system", "content": system_prompt})
        context.append({"role": "user", "content": text_obs})
    print("Environments and contexts initialized.")

    # Track episode statistics
    episode_stats = []
    current_episode_reward = [0.0] * num_envs
    current_episode_steps = [0] * num_envs
    current_episode_invalid_actions = [0] * num_envs

    # Run until desired number of episodes is reached
    while len(episode_stats) < num_episodes:
        print(f"Starting iteration for episode {len(episode_stats)+1}...")
        # Prepare queries for all environments
        query_tensors_list = [
            tokenizer.apply_chat_template(context, return_tensors="pt", add_generation_prompt=True).squeeze(0)
            for context in contexts
        ]

        # Use tokenizer's pad method, which will respect left-padding
        encoded = tokenizer.pad(
            {"input_ids": query_tensors_list},
            padding=True,
            return_tensors="pt"
        ).to(device)
        attention_mask = encoded["attention_mask"]

        # Generate actions using encoded["input_ids"]
        response_tensors_step = model.generate(
            encoded["input_ids"],
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        response_texts_step = tokenizer.batch_decode(response_tensors_step, skip_special_tokens=True)
        
        # Debug what the model is actually generating
        print("\n---- MODEL OUTPUTS ----")
        print(f"Valid actions: {list(utils.text_to_action.keys())}")
        for i, response in enumerate(response_texts_step):
            print(f"Raw output {i}: '{response}'")
            
        # Step through each environment
        for i, (env, response_text) in enumerate(zip(eval_env_managers, response_texts_step)):
            text_obs, reward, done = env.step(response_text)

            # Update episode trackers
            current_episode_reward[i] += reward
            current_episode_steps[i] += 1
            if response_text not in utils.text_to_action:
                current_episode_invalid_actions[i] += 1

            # Update context
            contexts[i].append({"role": "assistant", "content": response_text})
            contexts[i].append({"role": "user", "content": text_obs})

            if done:
                print(f"Episode finished in env {i}: steps={current_episode_steps[i]}, reward={current_episode_reward[i]}")
                # Record episode statistics
                success = 1 if reward > 0 else 0  # Success if final reward is positive
                episode_stats.append({
                    "success": success,
                    "total_reward": current_episode_reward[i],
                    "steps": current_episode_steps[i],
                    "invalid_actions": current_episode_invalid_actions[i],
                })

                # Reset episode trackers
                current_episode_reward[i] = 0.0
                current_episode_steps[i] = 0
                current_episode_invalid_actions[i] = 0

                # Reset environment and context
                mission, text_obs = env.reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_obs}
                ]

    # Compute metrics
    success_rate = np.mean([stat["success"] for stat in episode_stats])
    avg_reward = np.mean([stat["total_reward"] for stat in episode_stats])
    avg_steps = np.mean([stat["steps"] for stat in episode_stats])
    total_invalid_actions = sum(stat["invalid_actions"] for stat in episode_stats)
    total_actions = sum(stat["steps"] for stat in episode_stats)
    invalid_action_rate = total_invalid_actions / total_actions if total_actions > 0 else 0.0

    # Compute average steps for successful episodes
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

if __name__ == '__main__':
    print("Script started")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import AutoModelForCausalLMWithValueHead
    print("Imports successful")
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print("About to load tokenizer...")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Set generation parameters
    # In your __main__ section:
    generation_kwargs = {
        "max_new_tokens": 10,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    print("Generation kwargs set.")

    # Define evaluation settings
    env_id = "BabyAI-GoToObj-v0"
    num_eval_episodes = 10  # For debugging, use 10 episodes first
    num_envs = 4
    context_window = 5
    seed_offset = 1000

    print("Starting evaluation...")
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
    print("Evaluation completed.")
    print("Evaluation Metrics for Baseline Llama-3B-Instruct on the 'GoTo' Task:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")