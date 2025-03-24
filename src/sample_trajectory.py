import utils
import gym, babyai_text
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import Dict, List, Any, Tuple
from rich.pretty import pprint


def sample_trajectory(
    env: gym.Env,
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
        env (Env): The environment to sample from.
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
    
    obs, info = env.reset()
    done = False
    
    messages = []
    
    mission = obs["mission"]
    system_prompt = utils.get_system_prompt()
    system_prompt = system_prompt.replace("{goal}", mission)
    messages.append({"role": "system", "content": system_prompt})
    
    text_obs = "\n".join(info["descriptions"])
    messages.append({"role": "user", "content": text_obs})
    
    for step in range(max_steps):
        
        query_tensor = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device) # shape: (1, messages_length)
        query_tensor = query_tensor.squeeze(0) # shape: (messages_length,)
        query_tensors.append(query_tensor)
        
        # TODO: should return_prompt be True or False? According to the quickstart it should be False, however according to the PPO docs it should be True
        response_tensor = trainer.generate(query_tensor, **generation_kwargs, return_prompt=False) # shape: (1, new_tokens)
        response_tensor = response_tensor.squeeze(0) # shape: (new_tokens,)
        response_tensors.append(response_tensor)
        
        action_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": action_text})
        
        if action_text not in utils.text_to_action:
            invalid_action_message = "Invalid action, the valid actions are: " + ", ".join(utils.text_to_action.keys()) + ".\n"
            invalid_action_message += "Please output one of the above actions and nothing else."
            text_obs = invalid_action_message
            reward = -0.1
        else:
            action = utils.text_to_action[action_text]
            obs, reward, done, info = env.step(action)
            text_obs = "\n".join(info["descriptions"])
        
        reward = torch.tensor(reward).to(device)
        rewards.append(reward)
        messages.append({"role": "user", "content": text_obs})
        
        if done:
            final_reward = reward
            for reward in rewards[:-1]:
                reward += final_reward # add the final reward to all previous rewards
            break
        
    return query_tensors, response_tensors, rewards, messages, done


if __name__ == "__main__":
    
    env = gym.make("BabyAI-GoToObj-v0")
    
    config = PPOConfig(batch_size=4, mini_batch_size=4)
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
    ref_model = create_reference_model(model)
    
    trainer = PPOTrainer(config, model, ref_model, tokenizer)
    
    generation_kwargs = {
        "max_new_tokens": 10,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
    }
    
    observations, actions, rewards, messages, done = sample_trajectory(env, trainer, tokenizer, generation_kwargs, max_steps=100)
    pprint((observations, actions, rewards))
    pprint(messages)
    pprint(f"Done: {done}")
    pprint(f"Final reward: {rewards[-1]}")