import utils
from env_manager import EnvManager
import gym, babyai_text
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import Dict, List, Any, Tuple
import logging


def sample_batch(
        env_managers: List[EnvManager], 
        tokenizer: PreTrainedTokenizer, 
        trainer: PPOTrainer,
        generation_kwargs: Dict[str, Any],
        batch_size: int,
        logger: logging.Logger = None,
        context_window: int = 5, # Number of previous experiences to keep in context
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
    success_count, total_count = 0, 0
    Q, R, W = [], [], [] # Query, Response, and Reward tensors
    query_tensors_per_episode = [[] for _ in range(num_envs)]
    response_tensors_per_episode = [[] for _ in range(num_envs)]
    rewards_per_episode = [[] for _ in range(num_envs)]

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

    while len(W) < batch_size:

        query_tensors_step = []
        for context in contexts:
            if len(context) > (2 * context_window + 1):
                context = context[0:1] + context[-(2*context_window):]
            query_tensor = tokenizer.apply_chat_template(context, return_tensors="pt", add_generation_prompt=True).squeeze(0)
            query_tensors_step.append(query_tensor)
        
        response_tensors_step = trainer.generate(
            query_tensors_step,
            generation_kwargs=generation_kwargs,
            return_prompt=False,
        )
        response_texts_step = tokenizer.batch_decode(response_tensors_step, skip_special_tokens=True)
        
        for i, (env, response_text) in enumerate(zip(env_managers, response_texts_step)):

            query_tensors_per_episode[i].append(query_tensors_step[i])
            response_tensors_per_episode[i].append(response_tensors_step[i])

            text_obs, reward, done = env.step(response_text)
            rewards_per_episode[i].append(reward)
            contexts[i].append({"role": "assistant", "content": response_text})
            contexts[i].append({"role": "user", "content": text_obs})

            if i==0 and logger is not None:
                logger.info(f"ASSISTANT: {response_text}")
                logger.info(f"USER: {text_obs}")
                logger.info(f"REWARD: {reward} DONE: {done}")

            if done:
                # Discount future rewards if successful
                success = True if reward > 0 else False
                total_count += 1
                success_count += 1 if success else 0
                for j in range(len(rewards_per_episode[i])-1):
                    rewards_per_episode[i][j] += rewards_per_episode[i][-1]
                # Append trajectory to Q, R, W
                Q.extend(query_tensors_per_episode[i])
                R.extend(response_tensors_per_episode[i])
                W.extend(rewards_per_episode[i])
                # Reset env and contexts
                query_tensors_per_episode[i] = []
                response_tensors_per_episode[i] = []
                rewards_per_episode[i] = []
                mission, text_obs = env.reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text_obs}]
                if logger is not None:
                    logger.info(f"Environment {i} finished with success: {success}, resetting...")
                    logger.info("-"*20)
                    logger.info(f"SYSTEM: {contexts[i][0]['content']}")
                    logger.info(f"USER: {contexts[i][1]['content']}")
    
    # Convert rewards to tensors
    W = [torch.tensor(w, dtype=torch.float32) for w in W]
    
    # Compute stats
    success_rate = success_count / total_count if total_count > 0 else 0
    stats = {
        "success_rate": success_rate,
        "total_count": total_count,
        "success_count": success_count,
    }
    return Q, R, W, stats


if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(batch_size=4, mini_batch_size=4)
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
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
    context_window = 5

    num_envs = 1
    env_managers = [EnvManager(gym.make(env_id, seed=i)) for i in range(num_envs)]
    batch_size = 8

    start_time = time.time()
    Q, R, W, stats = sample_batch(env_managers, tokenizer, trainer, generation_kwargs, batch_size, context_window=context_window)
    elapsed_time = time.time() - start_time
    print(f"Success rate: {stats['success_rate']:.2f}")
    print(f"Success count: {stats['success_count']}")
    print(f"Total count: {stats['total_count']}")
    print(f"Sampling batch took {elapsed_time:.2f} seconds")