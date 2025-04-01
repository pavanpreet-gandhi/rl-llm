import utils
from env_manager import EnvManager
import gym, babyai_text
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import Dict, List, Any, Tuple
import logging
import time


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
    generate_times = []
    success_count_by_task = {task: 0 for task in utils.task_types}
    total_count_by_task = {task: 0 for task in utils.task_types}
    success_reward_by_task = {task: 0 for task in utils.task_types}
    episode_length_by_task = {task: 0 for task in utils.task_types}
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
        
        start_time = time.time()
        query_tensors_step = []
        for context in contexts:
            if len(context) > (2 * context_window + 1):
                context = context[0:1] + context[-(2*context_window)+1:]
            query_tensor = tokenizer.apply_chat_template(context, return_tensors="pt", add_generation_prompt=True).squeeze(0)
            query_tensors_step.append(query_tensor)
        
        response_tensors_step = trainer.generate(
            query_tensors_step,
            generation_kwargs=generation_kwargs,
            return_prompt=False,
        )
        response_texts_step = tokenizer.batch_decode(response_tensors_step, skip_special_tokens=True)
        generate_time = time.time() - start_time
        generate_times.append(generate_time)
        
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
                task = env.get_task()
                success = True if reward > 0 else False
                total_count_by_task[task] += 1
                success_count_by_task[task] += 1 if success else 0
                success_reward_by_task[task] += reward if success else 0
                episode_length = len(rewards_per_episode[i])
                episode_length_by_task[task] += episode_length
                if success:
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
                env.reset()
                mission, text_obs = env.reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text_obs}]
                if logger is not None:
                    logger.info(f"Environment {i} finished with success: {success} in {episode_length} steps")
                    logger.info("-"*20)
                    logger.info(f"SYSTEM: {contexts[i][0]['content']}")
                    logger.info(f"USER: {contexts[i][1]['content']}")
    
    # Convert rewards to tensors
    W = [torch.tensor(w, dtype=torch.float32) for w in W]
    
    # Compute stats
    total_count = sum(total_count_by_task.values())
    success_count = sum(success_count_by_task.values())
    success_rate = success_count / total_count if total_count > 0 else 0
    avg_success_reward = sum(success_reward_by_task.values()) / success_count if success_count > 0 else 0
    avg_episode_length = sum(episode_length_by_task.values()) / success_count if success_count > 0 else 0
    total_generate_time = sum(generate_times)
    min_generate_time = min(generate_times)
    max_generate_time = max(generate_times)
    stats = {
        "total_count": total_count,
        "success_rate": success_rate,
        "avg_success_reward": avg_success_reward,
        "avg_episode_length": avg_episode_length,
        "total_generate_time": total_generate_time,
        "min_generate_time": min_generate_time,
        "max_generate_time": max_generate_time,
    }
    success_rate_by_task = {task: success_count_by_task[task] / total_count_by_task[task] if total_count_by_task[task] > 0 else 0 for task in utils.task_types}
    avg_success_reward_by_task = {task: success_reward_by_task[task] / success_count_by_task[task] if success_count_by_task[task] > 0 else 0 for task in utils.task_types}
    avg_episode_length_by_task = {task: episode_length_by_task[task] / success_count_by_task[task] if success_count_by_task[task] > 0 else 0 for task in utils.task_types}
    for task in utils.task_types:
        stats[f"total_count_{task}"] = total_count_by_task[task]
        stats[f"success_rate_{task}"] = success_rate_by_task[task]
        stats[f"avg_success_reward_{task}"] = avg_success_reward_by_task[task]
        stats[f"avg_episode_length_{task}"] = avg_episode_length_by_task[task]

    return Q, R, W, stats


if __name__=="__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(batch_size=4, mini_batch_size=4)
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct" # "meta-llama/Llama-3.2-3B-Instruct"
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
    env_id = "BabyAI-MixedTrainLocal-v0"
    context_window = 3
    num_envs = 1
    batch_size = 8
    env_managers = [
        EnvManager(
            env_id, 
            invalid_action_penalty=-2,
            consecutive_invalid_actions_allowed=5,
        )
        for i in range(num_envs)
    ]

    start_time = time.time()
    Q, R, W, stats = sample_batch(env_managers, tokenizer, trainer, generation_kwargs, batch_size, context_window=context_window)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    from pprint import pprint
    pprint(f"Stats: {stats}")