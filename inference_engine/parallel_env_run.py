import gym
from babyai.paral_env_simple import ParallelEnv
import torch
import numpy as np
from copy import deepcopy
from torch import multiprocessing as mp
from torch.multiprocessing import Process, Pipe
from ppo_engine.sample_trajectory import sample_trajectory
import logging
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def multi_worker(conn, envs):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # generate_trajectories(trainer, tokenizer, generation_kwargs, device, max_steps)
        if cmd == "generate_trajectories":
            ret = []
            for env in envs:
                query_tensors, response_tensors, rewards, messages = sample_trajectory(
                    env, data[0], data[1], data[2], data[3], data[4]
                )
                ret.append((query_tensors, response_tensors, rewards, messages))
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

class ParallelTrajectory:
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
            self.envs_per_proc = 64
        else:
            raise NotImplementedError
        
        self.tokenizer = AutoTokenizer.from_pretrained(config_dict.model_id)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(config_dict.model_id).to(self.device)
        logger.info("Loaded model and tokenizer")
        
        self.ref_model = create_reference_model(self.model, num_shared_layers=config_dict.num_shared_layers)
        logger.info(f"Created reference model with {config_dict.num_shared_layers} shared layers")
        
        self.config = PPOConfig(
            batch_size=config_dict.batch_size, 
            mini_batch_size=config_dict.mini_batch_size
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

        # Spin up subprocesses
        self.locals = []
        self.processes = []
        self.start_processes()

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
        logger.info("done spinning up processes")
    
    def generate_trajectories(self):
        """Generate trajectories for all environments"""
        for i in range(0, self.n_parallel, self.envs_per_proc):
            self.locals[i // self.envs_per_proc].send(
                ("generate_trajectories", [self.trainer, self.tokenizer, self.generation_kwargs, self.device, self.max_steps])
            )
        ret_query_tensors, ret_response_tensors, ret_rewards, ret_messages = [], [], [], []
        for i in range(0, self.n_parallel, self.envs_per_proc):
            res = self.locals[i // self.envs_per_proc].recv()
            for j in range(len(res)):
                query_tensors, response_tensors, rewards, messages = res[j]
                ret_query_tensors += query_tensors
                ret_response_tensors += response_tensors
                ret_rewards += rewards
                ret_messages += messages
        return ret_query_tensors, ret_response_tensors, ret_rewards, ret_messages