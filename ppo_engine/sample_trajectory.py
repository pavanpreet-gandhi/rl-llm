import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import gym, babyai_text
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import Dict, List, Any, Tuple
from rich.pretty import pprint
import numpy as np
from inference_engine.babyai_text_env import BabyAITextEnv
from types import SimpleNamespace


def sample_trajectory(
    env: BabyAITextEnv,
    trainer: PPOTrainer,
    tokenizer: PreTrainedTokenizer,
    generation_kwargs: Dict[str, Any],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_steps: int = 128
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Dict[str, str]]]:
    """
    Sample a trajectory from the environment using actions sampled from the current policy of the PPO trainer.
    Backfill the rewards for each observation-action pair according to the final reward received.
    
    Args:
        env (Env): The environments to sample from.
        trainer (PPOTrainer): The PPO trainer containing the current policy.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding actions.
        generation_kwargs (Dict[str, Any]): Generation parameters for the policy.
        device (torch.device, optional): Device to use for computation. Defaults to GPU if available.
        max_steps (int, optional): Maximum number of steps to sample. Defaults to 128.
    
    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Dict[str, str]]]: 
        A tuple containing:
            - query_tensors: List of tensors representing the queries.
            - response_tensors: List of tensors representing the responses.
            - rewards: List of tensors representing the rewards.
            - messages: List of dictionaries containing the messages exchanged.
    """
    query_tensors, response_tensors, rewards = [], [], []
    
    obss, infos = env.reset()
    
    messages = []
    
    missions = [obs["mission"] for obs in obss]
    system_prompts = [utils.get_system_prompt().replace("{{goal}}", mission) for mission in missions]
    for system_prompt in system_prompts:
        messages.append({"role": "system", "content": system_prompt})
    
    text_obss = ["\n".join(info["descriptions"]) for info in infos]
    for text_obs in text_obss:
        messages.append({"role": "user", "content": text_obs})
    
    log_done_count = 0
    num_step = 0
    final_rewards = np.zeros(env.n_parallel)
    while num_step < max_steps or log_done_count < env.n_parallel:
        num_step += 1
        
        query_tensor = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device) # shape: (n_parallel, messages_length)
        query_tensors.append(query_tensor)
        
        # TODO: should return_prompt be True or False? According to the quickstart it should be False, however according to the PPO docs it should be True
        query_tensor_list = list(query_tensor.unbind(dim=0))
        response_tensor = trainer.generate(query_tensor_list, **generation_kwargs, return_prompt=False) # shape: (n_parallel, new_tokens)
        response_tensors.append(response_tensor)
        
        action_texts = tokenizer.batch_decode(response_tensor, skip_special_tokens=True)
        for action_text in action_texts:
            messages.append({"role": "assistant", "content": action_text})
        
            if action_text not in utils.text_to_action:
                text_obs = "You entered an invalid action, the valid actions are: " + str(list(utils.text_to_action.keys()))
                reward = -0.1
            else:
                actions = [utils.text_to_action[action_text] for action_text in action_texts]
                obss, rews, dones, infos = env.step(actions)
                text_obss = ["\n".join(info["descriptions"]) for info in infos]
        
        rews = torch.tensor(rews).to(device)
        rewards.append(rews)
        for text_obs in text_obss:
            messages.append({"role": "user", "content": text_obs})
        
        for i, done in enumerate(dones):
            if done:
                log_done_count += 1
                final_rewards[i] = rews[i]
                for reward in rewards[:-1]:
                    reward[i] += final_rewards[i] # add the final reward to all previous rewards
        
    return query_tensors, response_tensors, rewards, messages


if __name__ == "__main__":
    
    env_args = {
        "env_id": "BabyAI-MixedTrainLocal-v0",
        "seed": 42,
        "num_envs": 1,
        "action_space": utils.action_list,
    }
    env_args = SimpleNamespace(**env_args)
    env = BabyAITextEnv(env_args)
    
    config = PPOConfig(batch_size=4, mini_batch_size=4)
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
    ref_model = create_reference_model(model)
    
    trainer = PPOTrainer(config, model, ref_model, tokenizer)
    
    generation_kwargs = {
        "max_new_tokens": 20,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.8,
    }
    
    observations, actions, rewards, messages = sample_trajectory(env, trainer, tokenizer, generation_kwargs, max_steps=4)
    pprint((observations, actions, rewards))
    pprint(messages)