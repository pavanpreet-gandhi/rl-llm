import gym
import utils
from typing import Tuple, List
import random


class EnvManager:

    def __init__(
        self,
        env_ids: List[str], 
        invalid_action_penalty: float = -0.1, 
        consecutive_invalid_actions_allowed: int = 5, 
        reasoning_flag: bool = False
    ):
        self.env_ids = env_ids
        self.invalid_action_penalty = invalid_action_penalty
        self.consecutive_invalid_actions_allowed = consecutive_invalid_actions_allowed
        self.consecutive_invalid_actions = 0
        self.reasoning_flag = reasoning_flag
    
    def reset(self) -> Tuple[str, str]:
        self.env_id = random.choice(self.env_ids)
        self.env = gym.make(self.env_id, disable_env_checker=True)
        self.consecutive_invalid_actions = 0
        obs, info = self.env.reset()
        mission = obs["mission"]
        text_obs = "\n".join(info["descriptions"])
        return mission, text_obs
    
    def step(self, text_action: str) -> Tuple[str, float, bool]:
        if self.reasoning_flag:
            text_action = text_action.split("final answer:")[-1].strip()
            action = utils.text_to_action.get(text_action, None)
            if action is None:
                self.consecutive_invalid_actions += 1
                invalid_action_message = (
                    "Invalid format. Think step-by-step and end your response with 'final answer: [answer]', where [answer] is one of: "
                    + ", ".join(utils.text_to_action.keys())
                    + ".\n"
                )
                text_obs = invalid_action_message
                reward = self.invalid_action_penalty
                done = (
                    self.consecutive_invalid_actions
                    >= self.consecutive_invalid_actions_allowed
                )
            else:
                obs, reward, done, info = self.env.step(action)
                text_obs = "\n".join(info["descriptions"])

        else:
            action = utils.text_to_action.get(text_action, None)
            if action is None:
                self.consecutive_invalid_actions += 1
                invalid_action_message = (
                    "Invalid action, the valid actions are: "
                    + ", ".join(utils.text_to_action.keys())
                    + ".\n"
                )
                invalid_action_message += (
                    "Please output one of the above actions and nothing else."
                )
                text_obs = invalid_action_message
                reward = self.invalid_action_penalty
                done = (
                    self.consecutive_invalid_actions
                    >= self.consecutive_invalid_actions_allowed
                )
            else:
                obs, reward, done, info = self.env.step(action)
                text_obs = "\n".join(info["descriptions"])
        return text_obs, reward, done