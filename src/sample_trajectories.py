import torch
import gym, babyai_text
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import Dict, List, Any, Tuple
from rich.pretty import pprint
import utils
from env_manager import EnvManager

def sample_trajectories(
    env_managers: List[EnvManager],
    trainer: PPOTrainer,
    tokenizer: PreTrainedTokenizer,
    generation_kwargs: Dict[str, Any],
    device: torch.device,
    experiences_needed: int,
    max_steps_per_episode: int = 100,
    consecutive_invalid_actions_allowed: int = 5,
    invalid_action_penalty: float = -0.1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """
    Sample trajectories from multiple environments by performing batch inference with a single LLM.
    Continuously collect experiences until the specified number of experiences is reached.
    NOTE: This function can possibly return more than the specified number of experiences.
    TODO: Implement nice logging to easily spot any bugs or track training.

    Args:
        envs (List[EnvManager]): List of environment managers.
        trainer (PPOTrainer): PPO trainer instance.
        tokenizer (PreTrainedTokenizer): Tokenizer instance.
        generation_kwargs (Dict[str, Any]): Generation arguments for the LLM.
        experiences_needed (int): Number of experiences to collect.
        max_steps_per_episode (int): Maximum steps per episode.
        consecutive_invalid_actions_allowed (int): Number of consecutive invalid actions allowed before termination.
        invalid_action_penalty (float): Penalty for invalid actions.
    
    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]: Lists of query tensors, response tensors, and rewards.
    """
    # Initialize lists to store experiences
    num_envs = len(env_managers)
    query_tensors, response_tensors, rewards = [], [], []
    query_tensors_episode = [[] for _ in range(num_envs)]
    response_tensors_episode = [[] for _ in range(num_envs)]
    rewards_episode = [[] for _ in range(num_envs)]

    # Initialize contexts for the LLM
    contexts = [[] for _ in range(num_envs)]
    system_prompt_template = utils.get_system_prompt()
    missions, text_obss = zip(*[env.reset() for env in env_managers])
    for messages, mission, text_obs in zip(contexts, missions, text_obss):
        system_prompt = system_prompt_template.replace("{goal}", mission)
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text_obs})

    # Main loop to collect experiences
    while len(rewards) < experiences_needed:

        # Tokenize contexts
        query_tensors_step = tokenizer.apply_chat_template(
            contexts, 
            return_tensors="pt", 
            add_generation_prompt=True, 
            padding="longest",
            padding_side="left",
        ).to(device)
        query_tensors_step = [tensor for tensor in query_tensors_step] # Convert to list of tensors
        for i, tensor in enumerate(query_tensors_step):
            query_tensors_episode[i].append(tensor)

        # Generate responses
        response_tensors_step = trainer.generate(query_tensors_step, **generation_kwargs, return_prompt=False)
        for i, tensor in enumerate(response_tensors_step):
            response_tensors_episode[i].append(tensor)

        # Extract text actions
        action_texts = tokenizer.batch_decode(
            response_tensors_step, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Step through environments and collect rewards
        for i, (env, action_text) in enumerate(zip(env_managers, action_texts)):
            text_obs, reward, done, completed = env.step(action_text)
            contexts[i].append({"role": "assistant", "content": action_text})
            contexts[i].append({"role": "user", "content": text_obs})
            rewards_episode[i].append(reward)
            if len(rewards_episode[i]) > max_steps_per_episode:
                done = True

            # Reset environment if done
            if done:
                # If completed successfully, add final reward to all previous rewards
                if completed:
                    for j in range(len(rewards_episode[i])-1):
                        rewards_episode[i][j] += rewards_episode[i][-1]
                # Append experiences to main lists
                query_tensors.extend(query_tensors_episode[i])
                response_tensors.extend(response_tensors_episode[i])
                rewards.extend(rewards_episode[i])
                # Reset episode-specific lists
                query_tensors_episode[i] = []
                response_tensors_episode[i] = []
                rewards_episode[i] = []
                # Reset environment and context
                mission, text_obs = env.reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i] = [{"role": "system", "content": system_prompt}]
                contexts[i].append({"role": "user", "content": text_obs})
    
    # Convert rewards to tensors and return
    rewards = [torch.tensor(reward).to(device) for reward in rewards]
    return query_tensors, response_tensors, rewards


if __name__ == "__main__":
    # EXAMPLE USAGE

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(batch_size=4, mini_batch_size=4)
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
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
    env_managers = [EnvManager(gym.make("BabyAI-MixedTrainLocal-v0")) for _ in range(4)]

    # Sample trajectories
    query_tensors, response_tensors, rewards = sample_trajectories(
        env_managers,
        trainer,
        tokenizer,
        generation_kwargs,
        device,
        experiences_needed=32
    )

    # Print results
    for i, (query_tensor, response_tensor, reward) in enumerate(zip(query_tensors, response_tensors, rewards)):
        print(f"Experience {i}:")
        print(f"Query Tensor: {query_tensor.shape}")
        print(f"Response Tensor: {response_tensor.shape}")
        print(f"Reward: {reward}")
        print()