import gym
import utils
from typing import Tuple, List
import random


class EnvManager:

    def __init__(
        self,
        env_ids: List[str],
        num_dists: int = 0, # Number of distractors
        weights: List[float] = None,  # Sampling weights for env_ids
        invalid_action_penalty: float = -0.1, 
        consecutive_invalid_actions_allowed: int = 5, 
        reasoning_flag: bool = False
    ):
        self.env_ids = env_ids
        self.invalid_action_penalty = invalid_action_penalty
        self.num_dists = num_dists
        self.consecutive_invalid_actions_allowed = consecutive_invalid_actions_allowed
        self.consecutive_invalid_actions = 0
        self.reasoning_flag = reasoning_flag
        self.weights = weights
    
    def set_weights(self, weights: List[float]):
        self.weights = weights
    
    def reset(self) -> Tuple[str, str]:
        if self.weights is None or all(w == 0 for w in self.weights):
            self.env_id = random.choice(self.env_ids)
        else:
            self.env_id = random.choices(self.env_ids, weights=self.weights, k=1)[0]
        self.env = gym.make(self.env_id, disable_env_checker=True, num_dists=self.num_dists)
        self.consecutive_invalid_actions = 0
        obs, info = self.env.reset()
        mission = obs["mission"]
        text_obs = "\n".join(info["descriptions"])
        return mission, text_obs
    
    def step(self, text_action: str) -> Tuple[str, float, bool]:
        if self.reasoning_flag:
            issue_flag = True if "final answer:" not in text_action else False

            text_action = text_action.split("final answer:")[-1].strip()
            action = utils.text_to_action.get(text_action, None)
            if action is None or issue_flag:
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
                self.consecutive_invalid_actions = 0

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
                self.consecutive_invalid_actions = 0
        return text_obs, reward, done