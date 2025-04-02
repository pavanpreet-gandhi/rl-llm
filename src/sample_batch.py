import utils
from src import EnvManager
from typing import List, Union, Dict, Any
import time
import logging
import gym, babyai_text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead


def sample_batch(
        envs: List[EnvManager],
        tokenizer: AutoTokenizer,
        model: Union[AutoModelForCausalLM, AutoModelForCausalLMWithValueHead],
        generation_kwargs: Dict[str, Any],
        device: torch.device,
        batch_size: int,
        logger: logging.Logger = None,
        context_window: int = 5, 
        reasoning_flag: bool = False,
):
    """
    Sample a batch of experiences using the model from the given environments.
    """
    if logger is None:
        print("train logger is None")
    # Setup
    num_envs = len(envs)
    system_prompt_template = utils.get_system_prompt(reasoning_flag=reasoning_flag)

    # Initialize global storage lists
    queries_all, responses_all, rewards_all = [], [], []

    # Initialize per-environment storage lists
    queries_ep = [[] for _ in range(num_envs)]
    responses_ep = [[] for _ in range(num_envs)]
    rewards_ep = [[] for _ in range(num_envs)]

    # For train logger logging purposes
    txt_queries_ep = [[] for _ in range(num_envs)]
    txt_actions_ep = [[] for _ in range(num_envs)]


    # Reset envs and contexts
    contexts = [[] for _ in range(num_envs)] # each env has its own context represented as a list of system, user, and assistant messages
    missions, obss = zip(*[env.reset() for env in envs]) # reset all environments
    for i in range(num_envs):
        system_prompt = system_prompt_template.replace("{goal}", missions[i])
        contexts[i].append({"role": "system", "content": system_prompt})
        contexts[i].append({"role": "user", "content": obss[i]})

    # Variables to keep track of stats
    total_generate_time = 0

    # Main loop
    while len(rewards_all) < batch_size:
        
        # Time tokenization and generation
        start_time = time.time()

        # Clip contexts to the last context_window observations (keep the first system message)
        clipped_contexts = []
        for context in contexts:
            if len(context) > (2 * context_window + 1):
                clipped_context = [context[0]] + context[-(2*context_window)+1:]
            else:
                clipped_context = context
            clipped_contexts.append(clipped_context)

        # Create queries
        queries = tokenizer.apply_chat_template(
            clipped_contexts, 
            padding=True,
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(device)
        attention_mask = (queries != tokenizer.pad_token_id).long()

        # Generate responses
        output = model.generate(
            queries,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        responses = output[:, queries.shape[-1]:]

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
        # breakpoint()
        # Process each action sequentially
        for i in range(num_envs):
            # Store query and response
            queries_ep[i].append(queries[i])
            responses_ep[i].append(responses[i])

            txt_actions_ep[i].append(actions[i].strip())
            txt_queries_ep[i].append(tokenizer.decode(queries[i], skip_special_tokens=True).strip())

            # Take step in the environment
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
                
                # Discount rewards if successful
                if success:
                    for j in range(len(rewards_ep[i]) - 1):
                        rewards_ep[i][j] = final_reward # add final reward to all previous rewards
                
                # Append to global storage
                queries_all.extend(queries_ep[i])
                responses_all.extend(responses_ep[i])
                rewards_all.extend(rewards_ep[i])
                # Log query, response, and reward
                logger.info(f"Env {i}: Query: {txt_queries_ep[i]}, Response: {txt_actions_ep[i]}, Reward: {rewards_ep[i]}")

                # Reset environment, context, and per-environment storage
                queries_ep[i], responses_ep[i], rewards_ep[i], contexts[i] = [], [], [], []
                mission, text_obs = envs[i].reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i].append({"role": "system", "content": system_prompt})
                contexts[i].append({"role": "user", "content": text_obs})
    
    # Convert rewards to individual tensors
    rewards_all = [torch.tensor(reward, dtype=torch.float32).to(device) for reward in rewards_all]

    # Package stats
    stats = {
        "total_generate_time": total_generate_time,
    }

    return queries_all, responses_all, rewards_all, stats


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct" # "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    generation_kwargs = {
        "max_new_tokens": 20,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.8,
    }
    env_ids = ["BabyAI-GoTo-v0", "BabyAI-Pickup-v0"]
    context_window = 3
    num_envs = 1
    batch_size = 8
    env_managers = [
        EnvManager(
            env_ids, 
            invalid_action_penalty=-2,
            consecutive_invalid_actions_allowed=5,
        )
        for i in range(num_envs)
    ]

    start_time = time.time()
    queries, responses, rewards, stats = sample_batch(
        envs=env_managers,
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        model=model,
        generation_kwargs=generation_kwargs,
        device=device,
        batch_size=batch_size,
        context_window=context_window,
        reasoning_flag=False,
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    from pprint import pprint
    pprint(f"Stats: {stats}")
