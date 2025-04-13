import utils
from src import EnvManager
from typing import List, Union, Dict, Any
import time
import gym, babyai_text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def sample_episodes(
        envs: List[EnvManager],
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        generation_kwargs: Dict[str, Any],
        device: torch.device,
        number_of_episodes: int,
        context_window: int = 5, 
        reasoning_flag: bool = False,
):
    """
    Sample a batch of experiences using the model from the given environments.
    """
    # Setup
    num_envs = len(envs)
    system_prompt_template = utils.get_system_prompt(reasoning_flag=reasoning_flag)

    # Initialize per-environment storage lists
    rewards_ep = [[] for _ in range(num_envs)]

    # Reset envs and contexts
    episodes = 0
    contexts = [[] for _ in range(num_envs)] # each env has its own context represented as a list of system, user, and assistant messages
    missions, obss = zip(*[env.reset() for env in envs]) # reset all environments
    for i in range(num_envs):
        system_prompt = system_prompt_template.replace("{goal}", missions[i])
        contexts[i].append({"role": "system", "content": system_prompt})
        contexts[i].append({"role": "user", "content": obss[i]})

    # Variables to keep track of stats
    total_generate_time = 0
    possible_env_ids = envs[0].env_ids + ["all"] # assuming all envs have the same env_ids
    success_by_env_id = {env_id: [] for env_id in possible_env_ids}
    rewards_by_env_id = {env_id: [] for env_id in possible_env_ids}
    episode_lengths_by_env_id = {env_id: [] for env_id in possible_env_ids}
    num_invalid_actions_by_env_id = {env_id: [] for env_id in possible_env_ids}
    contexts_of_completed_episodes = []

    # Main loop
    while episodes < number_of_episodes:
        
        # Time tokenization and generation
        start_time = time.time()

        # Clip contexts and create query tensors
        queries = []
        for context in contexts:
            if len(context) > (2 * context_window + 1):
                clipped_context = [context[0]] + context[-(2 * context_window) + 1 :]
            else:
                clipped_context = context
            this_query = tokenizer.apply_chat_template(
                clipped_context, return_tensors="pt", add_generation_prompt=True
            ).squeeze(0)
            queries.append(this_query)

        # Generate responses
        output = model.generate(
            queries, generation_kwargs=generation_kwargs, return_prompt=False
        )
        responses = output

        # Extract actions
        actions = tokenizer.batch_decode(
            responses,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Time tokenization and generation
        end_time = time.time()
        generation_time = end_time - start_time
        total_generate_time += generation_time
        
        # Process each action sequentially
        for i in range(num_envs):
            # Take step in the environment and store reward
            text_obs, reward, done = envs[i].step(actions[i])
            rewards_ep[i].append(reward)

            # Update context
            contexts[i].append({"role": "assistant", "content": actions[i]})
            contexts[i].append({"role": "user", "content": text_obs})

            if done:
                # Collect stats
                final_reward = reward
                success = True if final_reward > 0 else False
                episode_length = len(rewards_ep[i])
                num_invalid_actions = sum([r < 0 for r in rewards_ep[i]])

                # Store stats
                env_id = envs[i].env_id
                for key in [env_id, "all"]:
                    success_by_env_id[key].append(1 if success else 0)
                    rewards_by_env_id[key].append(final_reward if success else 0)
                    episode_lengths_by_env_id[key].append(episode_length)
                    num_invalid_actions_by_env_id[key].append(num_invalid_actions)
                contexts_of_completed_episodes.append(contexts[i])
                
                # Discount rewards if successful
                if success:
                    for j in range(len(rewards_ep[i]) - 1):
                        rewards_ep[i][j] = final_reward # add final reward to all previous rewards

                # Reset environment, context, and per-environment storage
                rewards_ep[i], contexts[i] = [], []
                mission, text_obs = envs[i].reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i].append({"role": "system", "content": system_prompt})
                contexts[i].append({"role": "user", "content": text_obs})
                episodes += 1

                # Print progress
                print(episodes, end=" ")
    print()

    # Store stats
    stats = {}
    stats["total_generate_time"] = total_generate_time
    stats["success"] = success_by_env_id["all"]
    stats["rewards"] = rewards_by_env_id["all"]
    stats["episode_lengths"] = episode_lengths_by_env_id["all"]
    stats["num_invalid_actions"] = num_invalid_actions_by_env_id["all"]

    return stats, contexts


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation_kwargs = {
        "max_new_tokens": 20,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.8,
        "pad_token_id": tokenizer.pad_token_id,
    }
    env_ids = ["BabyAI-GoTo-v0", "BabyAI-Pickup-v0"]
    num_envs = 1
    env_managers = [
        EnvManager(
            env_ids, 
            invalid_action_penalty=-2,
            consecutive_invalid_actions_allowed=5,
        )
        for i in range(num_envs)
    ]
    print("Sampling batch...")
    start_time = time.time()
    stats = sample_episodes(
        envs=env_managers,
        tokenizer=tokenizer,
        model=model,
        generation_kwargs=generation_kwargs,
        device=device,
        number_of_episodes=5,
        context_window=5,
        reasoning_flag=False,
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    from rich.pretty import pprint
    pprint(f"Stats: {stats}")