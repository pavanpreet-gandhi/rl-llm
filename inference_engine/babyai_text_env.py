import gym
from babyai.paral_env_simple import ParallelEnv

class BabyAITextEnv(gym.Env):
    def __init__(self, config_dict):
        self.n_parallel = config_dict.num_envs
        self._action_space = [a.replace("_", " ") for a in config_dict.action_space]
        envs = []
        for i in range(config_dict.num_envs):
            env = gym.make(config_dict.env_id)
            env.seed(100 * config_dict.seed + i)
            envs.append(env)

        self._env = ParallelEnv(envs)

    def reset(self):
        obs, infos = self._env.reset()
        return obs, infos
    def step(self, actions_id, actions_command):
        obs, rews, dones, infos = self._env.step(actions_id)
        return obs, rews, dones, infos