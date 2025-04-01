import utils
from env_manager import EnvManager
import gym, babyai_text
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import Dict, List, Any, Tuple
import logging
import numpy as np
import time

def td_lambda_targets(rewards, next_values, gamma=0.99, lam=0.95):
    """
    Compute TD(λ) returns (targets) for one episode/trajectory of length T=N-1.

    Args:
        rewards     : 1D array of length T, where rewards[t] = r_{t+1} 
                      is the reward obtained when transitioning 
                      s_t -> s_{t+1}.
        next_values : 1D array of length T, where next_values[t] = V(s_{t+1}).
                      For a terminal s_{T}, you typically set next_values[T-1] = 0.
        gamma       : discount factor (0 <= gamma <= 1).
        lam         : the λ in TD(λ) (0 <= lam <= 1).

    Returns:
        targets     : 1D NumPy array of length T, where targets[t] 
                      is the TD(λ) return G_{λ}(t).
                      
    Notes:
    - If lam=0, this becomes 1-step TD:   G_t = r_{t+1} + gamma * V(s_{t+1}).
    - If lam=1, it accumulates future rewards all the way (plus final bootstrap),
      similar to a Monte Carlo return if the final is terminal and set to 0.
    - We iterate backwards so we can build each TD(λ) target with a single pass.
    """
    T = len(rewards)   # also len(next_values)
    targets = np.zeros(T, dtype=np.float32)

    G = 0.0  # Running 'backward' value for the λ-return
    for t in reversed(range(T)):
        # Recurrence for the λ-return:
        #   G = r_{t+1} + gamma * [ (1-lam)*V(s_{t+1}) + lam * G ]
        G = rewards[t] + gamma * ((1.0 - lam)*next_values[t] + lam*G)
        targets[t] = G

    return targets

def sample_batch(
        env_managers: List[EnvManager], 
        tokenizer: PreTrainedTokenizer, 
        trainer: PPOTrainer,
        generation_kwargs: Dict[str, Any],
        batch_size: int,
        logger: logging.Logger = None,
        context_window: int = 5 # Number of previous experiences to keep in 
    ) -> Tuple[List[torch.Tensor], List[float], List[torch.Tensor]]:
    """"
    Sample a batch of experiences from the environment.
    """
    # If logger is None create a temporary logger
    if logger is None:
        logger = logging.getLogger("sample_batch")
        logger.setLevel(logging.INFO)

    # Initialize variables
    num_envs = len(env_managers)
    successful_episode_count, total_episode_count, success_rewards, total_length = 0, 0, 0, 0
    total_generate_time, total_generate_count = 0, 0
    Q, R, W, D = [], [], [], [] # Query, Response, Reward and done tensors
    query_tensors_per_episode = [[] for _ in range(num_envs)]
    response_tensors_per_episode = [[] for _ in range(num_envs)]
    rewards_per_episode = [[] for _ in range(num_envs)]
    dones_per_episode = [[] for _ in range(num_envs)]

    # Reset envs and initialize contexts
    contexts = [[] for _ in range(num_envs)]
    system_prompt_template = utils.get_system_prompt()
    missions, text_obss = zip(*[env.reset() for env in env_managers])
    for i, (context, mission, text_obs) in enumerate(zip(contexts, missions, text_obss)):
        system_prompt = system_prompt_template.replace("{goal}", mission)
        context.append({"role": "system", "content": system_prompt})
        context.append({"role": "user", "content": text_obs})
        if i==0 and logger is not None:
            logger.info(f"SYSTEM: {context[0]['content']}")
            logger.info(f"USER: {context[1]['content']}")

    while total_length < batch_size:
        
        start_time = time.time()
        query_tensors_step = []
        for context in contexts:
            query_tensor = tokenizer.apply_chat_template(context, return_tensors="pt", add_generation_prompt=True).squeeze(0)
            query_tensors_step.append(query_tensor)
        
        # Generate responses for each environment
        response_tensors_step = trainer.generate(
            query_tensors_step,
            generation_kwargs=generation_kwargs,
            return_prompt=False,
        )
        # Decode response tensors to text
        response_texts_step = tokenizer.batch_decode(response_tensors_step, skip_special_tokens=True)
        generate_time = time.time() - start_time
        total_generate_time += generate_time
        total_generate_count += 1
        
        # Process each environment's response
        for i, (env, response_text) in enumerate(zip(env_managers, response_texts_step)):
            
            # Save query and response tensors
            query_tensors_per_episode[i].append(query_tensors_step[i])
            response_tensors_per_episode[i].append(response_tensors_step[i])

            # Take step in the environment and save reward
            text_obs, reward, done = env.step(response_text)
            rewards_per_episode[i].append(reward)
            dones_per_episode[i].append(done)

            # Update context with the new obs from the environment
            contexts[i].append({"role": "assistant", "content": response_text})
            contexts[i].append({"role": "user", "content": text_obs})
            # Clip context if necessary
            if len(context) > (2 * context_window + 1):
                context = context[0:1] + context[-(2*context_window):]

            if i==0 and logger is not None:
                logger.info(f"ASSISTANT: {response_text}")
                logger.info(f"USER: {text_obs}")
                logger.info(f"REWARD: {reward} | DONE: {done}")

            if done:
                # Track success
                success = True if reward > 0 else False
                total_episode_count += 1
                successful_episode_count += 1 if success else 0
                success_rewards += reward if success else 0
                episode_length = len(rewards_per_episode[i])
                total_length += episode_length
                # Append trajectory to Q, R, W
                Q.append(query_tensors_per_episode[i])
                R.append(response_tensors_per_episode[i])
                W.append(rewards_per_episode[i])
                D.append(dones_per_episode[i])
                # Reset env and contexts
                query_tensors_per_episode[i] = []
                response_tensors_per_episode[i] = []
                rewards_per_episode[i] = []
                dones_per_episode[i] = []
                mission, text_obs = env.reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text_obs}]
                if logger is not None:
                    logger.info(f"Environment {i} finished with success: {success} in {episode_length} steps")
                    logger.info("-"*20)
                    logger.info(f"SYSTEM: {contexts[i][0]['content']}")
                    logger.info(f"USER: {contexts[i][1]['content']}")
    
    # Compute stats
    success_rate = successful_episode_count / total_episode_count if total_episode_count > 0 else 0
    avg_success_reward = success_rewards / successful_episode_count if successful_episode_count > 0 else 0
    avg_generate_time = total_generate_time / total_generate_count if total_generate_count > 0 else 0
    stats = {
        "success_rate": success_rate,
        "total_count": total_episode_count,
        "success_count": successful_episode_count,
        "avg_success_reward": avg_success_reward,
        "avg_generate_time": avg_generate_time,
    }
    return Q, R, W, D, stats


if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(batch_size=4, mini_batch_size=4)
    model_id = "meta-llama/Llama-3.2-3B-Instruct" #"HuggingFaceTB/SmolLM2-135M-Instruct" # 
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    env_id = "BabyAI-GoToObj-v0" # "BabyAI-MixedTrainLocal-v0"
    context_window = 1

    num_envs = 4
    env_managers = [EnvManager(gym.make(env_id, seed=i)) for i in range(num_envs)]
    batch_size = 8

    start_time = time.time()
    Q, R, W, D, stats = sample_batch(env_managers, tokenizer, trainer, generation_kwargs, batch_size, context_window=context_window)
    elapsed_time = time.time() - start_time
    print(f"Success rate: {stats['success_rate']:.2f}")
    print(f"Success count: {stats['success_count']}")
    print(f"Total count: {stats['total_count']}")
    print(f"Sampling batch took {elapsed_time:.2f} seconds")
    print(W)