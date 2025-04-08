import utils
from src import EnvManager
from typing import List, Union, Dict, Any
import time
import logging
import gym, babyai_text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from TrajactoryPPOTrainer import log_memory, BatchedTrajectoryPPOTrainer
from trl import PPOTrainer, PPOConfig


class EpisodeCounter:
    def __init__(self):
        self.last_episode = 0
        self.batch_count = 0
        self.total_success_count = 0
        self.total_episode_count = 0
        self.batch_success_count = 0
        self.batch_episode_count = 0

    def increment(self, success: bool = False):
        if success:
            self.total_success_count += 1
            self.batch_success_count += 1
        self.total_episode_count += 1
        self.batch_episode_count += 1

    def new_batch(self):
        self.last_episode = self.total_episode_count
        self.batch_count += 1
        self.batch_success_count = 0
        self.batch_episode_count = 0

    def get_batch_success_rate(self):
        if self.batch_episode_count == 0:
            return 0
        return self.batch_success_count / self.batch_episode_count

    def get_total_success_rate(self):
        if self.total_episode_count == 0:
            return 0
        return self.total_success_count / self.total_episode_count


def sample_batch(
    envs: List[EnvManager],
    tokenizer: AutoTokenizer,
    trainer: PPOTrainer,
    generation_kwargs: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    logger: logging.Logger = None,
    context_window: int = 5,
    reasoning_flag: bool = False,
    trajectory_rl: bool = False,
    episode_counter: EpisodeCounter = None,
):
    """
    Sample a batch of experiences using the model from the given environments.
    """
    if logger is None:
        print("train logger is None")

    if tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "Cutting Knowledge Date: {{ cutoff }}\nToday Date: {{ today }}\n\n", ""
        )
    # Setup
    num_envs = len(envs)
    system_prompt_template = utils.get_system_prompt(reasoning_flag=reasoning_flag)
    episode_counter = episode_counter or EpisodeCounter()
    episode_counter.new_batch()
    # Initialize global storage lists
    queries_all, responses_all, rewards_all = [], [], []

    # Initialize per-environment storage lists
    queries_ep = [[] for _ in range(num_envs)]
    responses_ep = [[] for _ in range(num_envs)]
    rewards_ep = [[] for _ in range(num_envs)]
    dones_ep = [[] for _ in range(num_envs)]

    # For train logger logging purposes
    txt_queries_ep = [[] for _ in range(num_envs)]
    txt_actions_ep = [[] for _ in range(num_envs)]

    # Reset envs and contexts
    contexts = [
        [] for _ in range(num_envs)
    ]  # each env has its own context represented as a list of system, user, and assistant messages
    missions, obss = zip(*[env.reset() for env in envs])  # reset all environments
    for i in range(num_envs):
        system_prompt = system_prompt_template.replace("{goal}", missions[i])
        contexts[i].append({"role": "system", "content": system_prompt})
        contexts[i].append({"role": "user", "content": obss[i]})

    # Variables to keep track of stats
    total_generate_time = 0
    possible_env_ids = envs[0].env_ids + [
        "all"
    ]  # assuming all envs have the same env_ids
    success_by_env_id = {env_id: [] for env_id in possible_env_ids}
    rewards_by_env_id = {env_id: [] for env_id in possible_env_ids}
    episode_lengths_by_env_id = {env_id: [] for env_id in possible_env_ids}
    num_invalid_actions_by_env_id = {env_id: [] for env_id in possible_env_ids}

    # Main loop
    while len(rewards_all) < batch_size:

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
        output = trainer.generate(
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
            # Store query and response
            queries_ep[i].append(queries[i])
            responses_ep[i].append(responses[i])

            txt_actions_ep[i].append(actions[i].strip())
            txt_queries_ep[i].append(
                tokenizer.decode(queries[i], skip_special_tokens=True).strip()
            )

            # Take step in the environment
            text_obs, reward, done = envs[i].step(actions[i])
            rewards_ep[i].append(reward)
            dones_ep[i].append(done)

            # Update context
            contexts[i].append({"role": "assistant", "content": actions[i]})
            contexts[i].append({"role": "user", "content": text_obs})

            if done:
                # Collect stats
                final_reward = reward
                success = True if final_reward > 0 else False
                episode_counter.increment(success=success)
                episode_length = len(rewards_ep[i])
                num_invalid_actions = sum([r < 0 for r in rewards_ep[i]])

                # Store stats
                env_id = envs[i].env_id
                for key in [env_id, "all"]:
                    success_by_env_id[key].append(1 if success else 0)
                    rewards_by_env_id[key].append(final_reward if success else 0)
                    episode_lengths_by_env_id[key].append(episode_length)
                    num_invalid_actions_by_env_id[key].append(num_invalid_actions)

                if trajectory_rl and isinstance(trainer, BatchedTrajectoryPPOTrainer):
                    targets = trainer.compute_returns(
                        queries=queries_ep[i],
                        responses=responses_ep[i],
                        rewards=rewards_ep[i],
                        dones=dones_ep[i],
                    )
                    rewards_all.extend(targets)
                else:
                    # Discount rewards if successful
                    if success:
                        for j in range(len(rewards_ep[i]) - 1):
                            rewards_ep[i][
                                j
                            ] = final_reward  # add final reward to all previous rewards
                    rewards_all.extend(rewards_ep[i])

                # Append to global storage
                queries_all.extend(queries_ep[i])
                responses_all.extend(responses_ep[i])

                # Log query, response, and reward
                logger.info(
                    f"Env {i}: Query: {txt_queries_ep[i]}, Response: {txt_actions_ep[i]}, Reward: {rewards_ep[i]}"
                )

                # Reset environment, context, and per-environment storage
                (
                    queries_ep[i],
                    responses_ep[i],
                    rewards_ep[i],
                    dones_ep[i],
                    contexts[i],
                ) = ([], [], [], [], [])
                mission, text_obs = envs[i].reset()
                system_prompt = system_prompt_template.replace("{goal}", mission)
                contexts[i].append({"role": "system", "content": system_prompt})
                contexts[i].append({"role": "user", "content": text_obs})

    # Convert rewards to individual tensors
    rewards_all = [
        torch.tensor(reward, dtype=torch.float32).to(device) for reward in rewards_all
    ]

    # Package stats
    stats = {}
    stats["total_generate_time"] = total_generate_time
    for env_id in possible_env_ids:
        success_rate = (
            sum(success_by_env_id[env_id]) / len(success_by_env_id[env_id])
            if len(success_by_env_id[env_id]) > 0
            else 0
        )
        avg_reward = (
            sum(rewards_by_env_id[env_id]) / len(rewards_by_env_id[env_id])
            if len(rewards_by_env_id[env_id]) > 0
            else 0
        )
        avg_episode_length = (
            sum(episode_lengths_by_env_id[env_id])
            / len(episode_lengths_by_env_id[env_id])
            if len(episode_lengths_by_env_id[env_id]) > 0
            else 0
        )
        avg_invalid_actions = (
            sum(num_invalid_actions_by_env_id[env_id])
            / len(num_invalid_actions_by_env_id[env_id])
            if len(num_invalid_actions_by_env_id[env_id]) > 0
            else 0
        )
        stats[f"{env_id}_success_rate"] = success_rate
        stats[f"{env_id}_avg_reward"] = avg_reward
        stats[f"{env_id}_avg_episode_length"] = avg_episode_length
        stats[f"{env_id}_avg_invalid_actions"] = avg_invalid_actions
        stats[f"{env_id}_num_samples"] = len(success_by_env_id[env_id])
        stats["success_count_total"] = (episode_counter.total_success_count,)
        stats["episode_count_total"] = episode_counter.total_episode_count
        stats["total_success_rate"] = episode_counter.get_total_success_rate()

    return queries_all, responses_all, rewards_all, stats


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = (
        "meta-llama/Llama-3.2-3B-Instruct"  # "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    trainer = BatchedTrajectoryPPOTrainer(model=model)
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
        trainer=trainer,
        generation_kwargs=generation_kwargs,
        device=device,
        batch_size=batch_size,
        context_window=context_window,
        reasoning_flag=False,
        trajectory_rl=False,
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    from pprint import pprint

    pprint(f"Stats: {stats}")
