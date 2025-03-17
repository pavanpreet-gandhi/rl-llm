import gym
import torch
import numpy as np
from copy import deepcopy
from torch import multiprocessing as mp
from torch.multiprocessing import Process, Pipe
from ppo_engine.sample_trajectory import sample_trajectory
import logging
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import AutoTokenizer
import utils
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def multi_worker(conn, envs):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # step(actions, stop_mask, device)
        if cmd == "step":
            ret = []
            device = data[1]
            for env, a in zip(envs, data[0]):
                if a not in utils.text_to_action:
                    text_obs = "You entered an invalid action, the valid actions are: " + str(list(utils.text_to_action.keys()))
                    reward = -0.1
                    done = False
                    ret.append((None, reward, done, {"descriptions": text_obs}))
                else:
                    action = utils.text_to_action[a]
                    obs, reward, done, info = env.step(action)
                    ret.append((obs, reward, done, info))
            conn.send(ret)
        # reset()
        # data contains a list of bool values indicating whether to reset the env
        elif cmd == "reset":
            ret = []
            assert len(data) == len(envs), "Bool length does not match number of envs"
            for env, b in zip(envs, data):
                if b:
                    obs, info = env.reset()
                    ret.append((obs, info))
                else:
                    ret.append((None, None))
            conn.send(ret)
        # render_one()
        elif cmd == "render_one":
            mode, highlight = data
            ret = envs[0].render(mode, highlight)
            conn.send(ret)
            # __str__()
        elif cmd == "__str__":
            ret = str(envs[0])
            conn.send(ret)
        else:
            raise NotImplementedError

class ParallelTrainer:
    def __init__(self, config_dict):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        self.n_parallel = config_dict.num_envs
        self.action_space = [a.replace("_", " ") for a in config_dict.action_space]
        envs = []
        for i in range(config_dict.num_envs):
            env = gym.make(config_dict.env_id)
            env.seed(100 * config_dict.seed + i)
            envs.append(env)

        self.envs = envs
        self.spec = deepcopy(self.envs[0].unwrapped.spec)
        self.spec_id = f"ParallelShapedEnv<{self.spec.id}>"
        self.env_name = self.envs[0].unwrapped.spec.id
        logger.info(f"Created {self.n_parallel} environments with id: {self.env_name} and seed: {config_dict.seed}")

        if "BabyAI" in self.env_name:
            self.envs_per_proc = 4
        else:
            raise NotImplementedError
        
        self.tokenizer = AutoTokenizer.from_pretrained(config_dict.model_id, padding_side="left")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(config_dict.model_id).to(self.device)
        logger.info("Loaded model and tokenizer")
        
        self.ref_model = create_reference_model(self.model, num_shared_layers=config_dict.num_shared_layers)
        logger.info(f"Created reference model with {config_dict.num_shared_layers} shared layers")
        
        self.config = PPOConfig(
            batch_size=config_dict.batch_size, 
            mini_batch_size=config_dict.mini_batch_size,
            ppo_epochs=config_dict.epochs
        )
        self.trainer = PPOTrainer(self.config, self.model, self.ref_model, self.tokenizer)
        logger.info("Initialized PPO Trainer")
        
        self.generation_kwargs = {
            "max_new_tokens": config_dict.max_new_tokens,
            "do_sample": config_dict.do_sample,
            "top_k": config_dict.top_k,
            "top_p": config_dict.top_p,
            "temperature": config_dict.temperature,
        }
        logger.info("Set up generation kwargs")

        self.max_steps = config_dict.max_steps_env
        self.num_steps_train = config_dict.num_steps_train
        self.query_tensors, self.response_tensors, self.rewards, self.dones, self.messages = [], [], [], [], []

        # Setup arrays to hold current observation
        # for each environment
        self.obss = []

        # Spin up subprocesses
        self.locals = []
        self.processes = []
        self.num_env_processes = []
        self.start_processes()
        self.reset()
        self.memory_size = config_dict.memory_size

    def __len__(self):
        return self.n_parallel

    def __str__(self):
        self.locals[0].send(("__str__", None))
        return f"<ParallelShapedEnv<{self.locals[0].recv()}>>"

    def __del__(self):
        for p in self.processes:
            p.terminate()

    def render(self, mode="rgb_array", highlight=False):
        """Render a single environment"""
        if "BabyAI" in self.spec_id:
            self.locals[0].send(("render_one", (mode, highlight)))
        else:
            raise NotImplementedError
        return self.locals[0].recv()

    def start_processes(self):
        """Spin up the n_parallel/envs_per_proc number of processes"""
        logger.info(f"spinning up {self.n_parallel} processes")
        for i in range(0, self.n_parallel, self.envs_per_proc):
            local, remote = Pipe()
            self.locals.append(local)
            if "BabyAI" in self.spec_id:
                p = Process(target=multi_worker,
                            args=(remote, self.envs[i:i + self.envs_per_proc]))
            else:
                raise NotImplementedError
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)
            self.num_env_processes.append(len(self.envs[i:i + self.envs_per_proc]))
        logger.info("done spinning up processes")

    def request_reset_envs(self):
        """Request all processes to reset their envs"""
        logger.info("requesting resets")
        for i, local in enumerate(self.locals):
            local.send(("reset", [True for _ in range(self.num_env_processes[i])]))
        self.obss = []
        logger.info("requested resets")

        infos = []
        for local in self.locals:
            res = local.recv()

            for j in range(len(res)):
                infos.append(res[j][1])
                if res[j][0] is not None:
                    self.obss += [res[j][0]]
        logger.info("completed resets")
        return infos

    def reset(self):
        """Reset all environments"""
        infos = self.request_reset_envs()
        obs = [ob for ob in self.obss]
        self.missions = [ob["mission"] for ob in obs]
        system_prompt = utils.get_system_prompt()
        system_prompts = [system_prompt.replace("{goal}", mission) for mission in self.missions]
        self.messages = [[{"role": "system", "content": system_prompt}] for system_prompt in system_prompts]
        self.mission_messages = self.messages.copy()
        for i, sub_message in enumerate(self.messages):
            sub_message.append({"role": "user", "content": "\n".join(infos[i]["descriptions"])})
    
    def reset_some(self, ids):
        """Reset some environment"""
        if not ids:
            return
        process_ids = [i // self.envs_per_proc for i in ids]
        reset_flags = [i % self.envs_per_proc for i in ids]
        reset_dict = {i: [False for j in range(self.num_env_processes[i])] for i in process_ids}
        system_prompt = utils.get_system_prompt()
        for i, reset_flag in zip(process_ids, reset_flags):
            reset_dict[i][reset_flag] = True
        for proc_id, flags in reset_dict.items():
            self.locals[proc_id].send(("reset", flags))
            res = self.locals[proc_id].recv()
            for idx, (obs_item, info_item) in enumerate(res):
                if flags[idx]:
                    this_id = proc_id * self.envs_per_proc + idx
                    self.obss[this_id] = obs_item
                    self.missions[this_id] = self.obss[this_id]["mission"]
                    self.messages[this_id] = [{"role": "system", "content": system_prompt.replace("{goal}", self.missions[this_id])}]
                    self.mission_messages[this_id] = self.messages[this_id].copy()
                    self.messages[this_id].append({"role": "user", "content": "\n".join(info_item["descriptions"])})
    
    def clear_memory(self):
        self.query_tensors = []
        self.response_tensors = []
        self.rewards = []
        self.dones = []

    def request_step(self, actions):
        """Request processes to step corresponding to (primitive) actions
           unless stop mask indicates otherwise"""
        for i in range(0, self.n_parallel, self.envs_per_proc):
            self.locals[i // self.envs_per_proc].send(
                ("step", [actions[i:i + self.envs_per_proc], self.device])
            )
        results = []
        for i in range(0, self.n_parallel, self.envs_per_proc):
            res = self.locals[i // self.envs_per_proc].recv()
            for j in range(len(res)):
                results.append(res[j])
                if results[-1][0] != None:
                    self.obss[i + j] = results[-1][0]
        return zip(*results)

    def step(self, action_texts):
        """Complete a step and evaluate low-level policy / termination
           classifier as needed depending on reward shaping scheme.
           
           Returns:  obs: list of environment observations,
                     reward: list of extrinsic rewards,
                     done: list of booleans,
                     info: depends on self.reward_shaping. Output can be used
                           to shape the reward.
        """
        # Make sure input is numpy array
        if type(action_texts) != np.ndarray:
            if type(action_texts) == list or type(action_texts) == str:
                action_texts = np.array(action_texts)
            elif type(action_texts) == torch.Tensor:
                action_texts = action_texts.cpu().numpy()
            else:
                raise TypeError
        actions_to_take = action_texts.copy()

        # Make a step in the environment
        obs, reward, done, info = self.request_step(actions_to_take)

        return obs, reward, done, info
    
    def extract_history(self):
        # extract last memory_size pair of messages
        # m[0] is the mission message
        return [[m[0]] + m[-min(2 * self.memory_size, len(m) - 1):] for m in self.messages]

    def generate_actions(self):
        query_tensors = self.tokenizer.apply_chat_template(self.extract_history(), return_tensors='pt', add_generation_prompt=True, padding='max_length', max_length=1024, tokenizer_kwargs={'padding_side' : 'left'}, truncation=True).to(self.device)
        self.query_tensors.append(query_tensors)

        response_tensors = self.trainer.generate(list(query_tensors), **self.generation_kwargs, return_prompt=False)
        response_tensors_id_dict_list = []
        for i in range(len(response_tensors)):
            response_tensors_id_dict_list.append({"input_ids": response_tensors[i]})
        padded_response_tensors = self.tokenizer.pad(response_tensors_id_dict_list, return_tensors='pt', padding='max_length', padding_side='left', max_length=self.generation_kwargs['max_new_tokens'])['input_ids'].to(self.device)
        self.response_tensors.append(padded_response_tensors)

        action_texts = self.tokenizer.batch_decode(padded_response_tensors)
        for i, sub_message in enumerate(self.messages):
            sub_message.append({"role": "assistant", "content": action_texts[i]})
        return action_texts
    
    def generate_trajectories(self):
        """Generate trajectories for all environments"""
        self.clear_memory()
        for i in range(self.max_steps):
            action_texts = self.generate_actions()
            obs, reward, done, info = self.step(action_texts)
            self.rewards.append(torch.tensor(reward).to(self.device))
            self.dones.append(torch.tensor(done))
            for im, m in enumerate(self.messages):
                if reward[im] == -0.1:
                    m.append({"role": "user", "content": info[im]["descriptions"]})
                else:
                    m.append({"role": "user", "content": "\n".join(info[im]["descriptions"])})
            self.reset_some([j for j, d in enumerate(done) if d])
        return self.query_tensors, self.response_tensors, self.rewards
    
    def collect_batch(self):
        """Collect a batch of trajectories"""
        batch_query_tensors, batch_response_tensors, batch_rewards = [], [], []
        for i in range(self.config.batch_size // self.n_parallel // self.max_steps):
            query_tensors, response_tensors, rewards = self.generate_trajectories()
            batch_query_tensors.extend(list(torch.cat(list(torch.stack(query_tensors, dim=1)), dim=0)))
            batch_response_tensors.extend(list(torch.cat(list(torch.stack(response_tensors, dim=1)), dim=0)))
            batch_rewards.extend(list(torch.cat(list(torch.stack(rewards, dim=1)), dim=0)))
        return batch_query_tensors, batch_response_tensors, batch_rewards
    
    def train(self):
        """Train the model"""
        logger.info("Starting parallel training loop")
        for i in tqdm(range(self.num_steps_train)):
            logger.info("Collecting experiences")
            query_tensors, response_tensors, rewards = self.collect_batch()
            stats = self.trainer.step(query_tensors, response_tensors, rewards)
            self.trainer.log_stats(stats, {"query": query_tensors, "response": response_tensors}, rewards, columns_to_log=["reward_mean", "reward_std", "objective/kl", "ppo/policy_loss", "ppo/value_loss"])
            logger.info(f"Training step {i} completed")
        return stats