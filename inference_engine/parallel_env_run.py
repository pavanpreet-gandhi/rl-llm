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
import utils
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def multi_worker(conn, envs):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # step(actions, stop_mask, device)
        if cmd == "step":
            ret = []
            device = data[2]
            for env, a, stopped in zip(envs, data[0], data[1]):
                if a not in utils.text_to_action:
                    text_obs = "You entered an invalid action, the valid actions are: " + str(list(utils.text_to_action.keys()))
                    reward = -0.1
                    done = False
                    ret.append((None, reward, done, {"descriptions": text_obs}))
                elif not stopped:
                    action = utils.text_to_action[a]
                    obs, reward, done, info = env.step(action)
                    reward = torch.tensor(reward).to(device)
                    if done:
                        obs, info = env.reset()
                    ret.append((obs, reward, done, info))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env in envs:
                obs, info = env.reset()
                ret.append((obs, info))
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(config_dict.model_id, padding_side="left")
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
        self.query_tensors, self.response_tensors, self.rewards, self.messages = [], [], [], []

        # Setup arrays to hold current observation and timestep
        # for each environment
        self.obss = []
        self.ts = np.array([0 for _ in range(self.n_parallel)])

        # Spin up subprocesses
        self.locals = []
        self.processes = []
        self.start_processes()
        init_obs, init_info = self.reset()
        self.missions = [obs["mission"] for obs in init_obs]
        system_prompt = utils.get_system_prompt()
        system_prompts = [system_prompt.replace("{goal}", mission) for mission in self.missions]
        self.messages = [[{"role": "system", "content": system_prompt}] for system_prompt in system_prompts]
        for i, sub_message in enumerate(self.messages):
            sub_message.append({"role": "user", "content": "\n".join(init_info[i]["descriptions"])})

        self.done_mask = np.array([False for _ in range(self.n_parallel)])

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

    def request_reset_envs(self):
        """Request all processes to reset their envs"""
        logger.info("requesting resets")
        for local in self.locals:
            local.send(("reset", None))
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
        return [obs for obs in self.obss], infos

    def request_step(self, actions, stop_mask):
        """Request processes to step corresponding to (primitive) actions
           unless stop mask indicates otherwise"""
        for i in range(0, self.n_parallel, self.envs_per_proc):
            self.locals[i // self.envs_per_proc].send(
                ("step", [actions[i:i + self.envs_per_proc],
                          stop_mask[i:i + self.envs_per_proc], self.device])
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
                     reward: np.array of extrinsic rewards,
                     done: np.array of booleans,
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
        stop_mask = np.array([False for _ in range(self.n_parallel)])
        obs, reward, done, info = self.request_step(actions_to_take, stop_mask)
        reward = np.array(reward)
        done_mask = np.array(done)

        self.ts += 1
        self.ts[done_mask] *= 0

        return [obs for obs in self.obss], reward, done_mask, info

    def generate_actions(self):
        query_tensors = self.tokenizer.apply_chat_template(self.messages, return_tensors='pt', add_generation_prompt=True, padding='max_length', max_length=512, truncation=True).to(self.device)
        self.query_tensors.append(query_tensors)

        response_tensors = self.trainer.generate(list(query_tensors), **self.generation_kwargs, return_prompt=False)
        response_tensors_id_dict_list = []
        for i in range(len(response_tensors)):
            response_tensors_id_dict_list.append({"input_ids": response_tensors[i]})
        padded_response_tensors = self.tokenizer.pad(response_tensors_id_dict_list, return_tensors='pt', padding='max_length', max_length=self.generation_kwargs['max_new_tokens'])['input_ids']
        self.response_tensors.append(padded_response_tensors)

        action_texts = self.tokenizer.batch_decode(padded_response_tensors)
        for i, sub_message in enumerate(self.messages):
            sub_message.append({"role": "assistant", "content": action_texts[i]})
        return action_texts
    
    def generate_trajectories(self):
        """Generate trajectories for all environments"""
        for i in range(self.max_steps):
            if all([done for done in self.done_mask]):
                break
            action_texts = self.generate_actions()
            obs, reward, done, info = self.step(action_texts)
            self.rewards.append(reward)
            for im, m in enumerate(self.messages):
                if reward[im] == -0.1:
                    m.append({"role": "user", "content": info[im]["descriptions"]})
                else:
                    m.append({"role": "user", "content": "\n".join(info[im]["descriptions"])})
            for j, this_done in enumerate(done):
                if this_done:
                    self.done_mask[j] = True
                    final_reward = reward[j]
                    for k in range(i-1):
                        self.rewards[k][j] += final_reward
        return self.query_tensors, self.response_tensors, self.rewards, self.messages