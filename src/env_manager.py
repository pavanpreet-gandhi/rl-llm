import gym
import utils
from typing import Tuple, List
import random
import logging
import re

# Set up logging
logging.basicConfig(
    filename="outputs/logs/env_manager.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        self.env = gym.make(self.env_id, disable_env_checker=True, num_dists=0)
        self.consecutive_invalid_actions = 0
        obs, info = self.env.reset()
        mission = obs["mission"]
        text_obs = "\n".join(info["descriptions"])
        logging.info(f"Reset env {self.env_id} with mission: {mission}")
        return mission, text_obs
    
    def step(self, text_action: str) -> Tuple[str, float, bool]:
        logging.debug(f"Raw text_action (reasoning={self.reasoning_flag}): '{text_action}'")
        
        if self.reasoning_flag:
            # Case-insensitive search for "final answer:"
            match = re.search(r"final answer:\s*(.*)", text_action, re.IGNORECASE)
            if match:
                extracted_action = match.group(1).strip()
                logging.debug(f"Extracted action after 'final answer:': '{extracted_action}'")
            else:
                extracted_action = text_action.strip()
                logging.debug(f"No 'final answer:' found, using: '{extracted_action}'")
            
            action = utils.text_to_action.get(extracted_action, None)
            logging.debug(f"Mapped action: {action}")
            
            if action is None:
                self.consecutive_invalid_actions += 1
                logging.warning(
                    f"Invalid action: '{extracted_action}'. "
                    f"Consecutive invalid actions: {self.consecutive_invalid_actions}"
                )
                invalid_action_message = (
                    "Invalid format or action. Think step-by-step and end your response with 'final answer: [answer]', where [answer] is one of: "
                    + ", ".join(utils.text_to_action.keys())
                    + ".\n"
                )
                text_obs = invalid_action_message
                reward = self.invalid_action_penalty
                done = (
                    self.consecutive_invalid_actions
                    >= self.consecutive_invalid_actions_allowed
                )
                logging.debug(f"Returning penalty: {reward}, Done: {done}")
            else:
                obs, reward, done, info = self.env.step(action)
                text_obs = "\n".join(info["descriptions"])
                logging.debug(f"Valid action executed: {action}, Reward: {reward}, Done: {done}")
        else:
            extracted_action = text_action.strip()
            action = utils.text_to_action.get(extracted_action, None)
            logging.debug(f"Non-reasoning action: '{extracted_action}', Mapped action: {action}")
            
            if action is None:
                self.consecutive_invalid_actions += 1
                logging.warning(
                    f"Invalid non-reasoning action: '{extracted_action}'. "
                    f"Consecutive invalid actions: {self.consecutive_invalid_actions}"
                )
                invalid_action_message = (
                    "Invalid action, the valid actions are: "
                    + ", ".join(utils.text_to_action.keys())
                    + ".\n"
                    "Please output one of the above actions and nothing else."
                )
                text_obs = invalid_action_message
                reward = self.invalid_action_penalty
                done = (
                    self.consecutive_invalid_actions
                    >= self.consecutive_invalid_actions_allowed
                )
                logging.debug(f"Returning penalty: {reward}, Done: {done}")
            else:
                obs, reward, done, info = self.env.step(action)
                text_obs = "\n".join(info["descriptions"])
                logging.debug(f"Valid non-reasoning action executed: {action}, Reward: {reward}, Done: {done}")
        
        return text_obs, reward, done