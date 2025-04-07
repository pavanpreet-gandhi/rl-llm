# Import necessary libraries
import numpy as np
import torch
import wandb
import gym
import matplotlib.pyplot as plt
from env_manager import EnvManager
import babyai_text
import utils
from peft import PeftModel, PeftConfig
from evaluate2 import evaluate_models
from typing import Dict, Any, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
babyai_text.register_levels(__name__, globals())



# Evaluation function to evaluate a model on a specific environment
def evaluate(
    model: torch.nn.Module, # The PyTorch model to evaluate
    tokenizer, # Tokenizer for processing input
    generation_kwargs: Dict[str, Any], # Dictionary of generation parameters
    num_episodes: int, # Number of episodes to evaluate
    env_id: str, # The environment ID
    env_kwargs: Dict[str, Any] = {}, # Additional environment arguments
    num_envs: int = 4, # Number of parallel environments
    context_window: int = 5, # Context window size
    seed_offset: int = 1000, # Offset for random seed
) -> Dict[str, float]: # Dictionary to store evaluation metrics
    
    print(f"Evaluating {env_id} with {num_episodes} episodes...")

    # Use cuda if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Function to evaluate multiple models on multiple environments
def evaluate_models(
    models_info: List[Dict[str, Any]],
    env_ids: List[str],
    num_episodes: int = 10,
    generation_kwargs: Dict[str, Any] = None,
    env_kwargs: Dict[str, Any] = {},
    num_envs: int = 4,
    context_window: int = 5,
    seed_offset: int = 1000,
    step: int = None,
    log_to_wandb: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": 3,
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
    results = {}
    
    for model_info in models_info:
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        model_name = model_info['name']
        
        print(f"\n=== Evaluating model: {model_name} ===")
        model_results = {}
        
        for env_id in env_ids:
            print(f"  Evaluating on environment: {env_id}")
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                generation_kwargs=generation_kwargs,
                num_episodes=num_episodes,
                env_id=env_id,
                env_kwargs=env_kwargs,
                num_envs=num_envs,
                context_window=context_window,
                seed_offset=seed_offset,
            )
            model_results[env_id] = metrics
            
            # Log to wandb if requested
            if log_to_wandb:
                for metric_name, value in metrics.items():
                    wandb.log({
                        f"eval/{model_name}/{env_id}/{metric_name}": value
                    }, step=step)
                    
            # Print results
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")
                
        results[model_name] = model_results
    
    return results

# Example usage

# Initialize wandb (optional)
wandb.init(project="babyai-ppo-evaluation", name="peft-model-eval")

# Load model
checkpoint = "pavanpreet-gandhi/babyai-ppo-2025-03-30_11-36-26"
peft_config = PeftConfig.from_pretrained(checkpoint)

# Get tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path, 
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model and apply PEFT adapter
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(base_model, checkpoint)
model.eval()

# Define evaluation environment(s)
env_ids = ["BabyAI-GoToObj-v0", "BabyAI-Pickup-v0"]

# Configure generation parameters
generation_kwargs = {
    "max_new_tokens": 20,
    "do_sample": True,
    "top_k": 10,
    "top_p": 0.95,
    "temperature": 0.8
}

# Prepare model info
models_info = [{
    'model': model,
    'tokenizer': tokenizer,
    'name': 'babyai-ppo'
}]

# Run evaluation
results = evaluate_models(
    models_info=models_info,
    env_ids=env_ids,
    num_episodes=50,  # More episodes for robust evaluation
    generation_kwargs=generation_kwargs,
    log_to_wandb=True,
    step=0  # Use appropriate step if tracking training progress
)

# Print summary
print("\n=== Evaluation Summary ===")
for model_name, env_results in results.items():
    for env_id, metrics in env_results.items():
        print(f"{model_name} on {env_id}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")