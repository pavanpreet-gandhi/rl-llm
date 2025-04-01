import gym
import utils
from typing import Tuple

class EnvManager:

    def __init__(self, env: gym.Env, invalid_action_penalty: float = -0.1, consecutive_invalid_actions_allowed: int = 5):
        self.env = env
        self.invalid_action_penalty = invalid_action_penalty
        self.consecutive_invalid_actions_allowed = consecutive_invalid_actions_allowed
        self.consecutive_invalid_actions = 0
        self.task = None
    
    def reset(self) -> Tuple[str, str]:
        self.consecutive_invalid_actions = 0
        obs, info = self.env.reset()
        mission = obs["mission"]
        self.task = utils.get_task_from_mission(mission)
        text_obs = "\n".join(info["descriptions"])
        return mission, text_obs
    
    def get_task(self) -> str:
        return self.task
    
    def step(self, text_action: str) -> Tuple[str, float, bool]:
        action = utils.text_to_action.get(text_action, None)
        if action is None:
            self.consecutive_invalid_actions += 1
            invalid_action_message = "Invalid action, the valid actions are: " + ", ".join(utils.text_to_action.keys()) + ".\n"
            invalid_action_message += "Please output one of the above actions and nothing else."
            text_obs = invalid_action_message
            reward = self.invalid_action_penalty
            done = self.consecutive_invalid_actions >= self.consecutive_invalid_actions_allowed
        else:
            obs, reward, done, info = self.env.step(action)
            text_obs = "\n".join(info["descriptions"])
        return text_obs, reward, done