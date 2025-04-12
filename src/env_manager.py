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
        weights: List[float] = None,  # Added weights parameter
        invalid_action_penalty: float = -0.1, 
        consecutive_invalid_actions_allowed: int = 5, 
        reasoning_flag: bool = False
    ):
        self.env_ids = env_ids
        self.weights = weights  # Save weights
        self.invalid_action_penalty = invalid_action_penalty
        self.consecutive_invalid_actions_allowed = consecutive_invalid_actions_allowed
        self.consecutive_invalid_actions = 0
        self.reasoning_flag = reasoning_flag
        self.last_observation = ""
    
    def set_weights(self, weights: List[float]):
        self.weights = weights
        logging.info(f"Updated weights: {self.weights}")

    def reset(self) -> Tuple[str, str]:
        # Use weights if provided and non-zero, else use uniform random selection
        if self.weights is None or all(w == 0 for w in self.weights):
            self.env_id = random.choice(self.env_ids)
            logging.debug("Using uniform random selection for environment ID")
        else:
            self.env_id = random.choices(self.env_ids, weights=self.weights, k=1)[0]
            logging.debug(f"Using weighted selection for environment ID: {self.env_id}")

        self.env = gym.make(self.env_id, disable_env_checker=True, num_dists=0)
        self.consecutive_invalid_actions = 0
        obs, info = self.env.reset()
        mission = obs["mission"]
        text_obs = "\n".join(info["descriptions"])
        self.last_observation = text_obs
        logging.info(f"Reset env {self.env_id} with mission: {mission}")
        return mission, text_obs
    
    def step(self, text_action: str) -> Tuple[str, float, bool]:
        logging.debug(f"Raw text_action (reasoning={self.reasoning_flag}): '{text_action}'")
        
        # Action variation mapping
        action_variations = {
            "move forward": "go forward",
            "forward": "go forward",
            "move left": "turn left",
            "left": "turn left",
            "move right": "turn right",
            "right": "turn right",
            "pickup": "pick up",
            "choose to turn left": "turn left",
            "choose to turn right": "turn right",
            "i should turn left": "turn left",
            "i should turn right": "turn right",
            "turn left now": "turn left",
            "turn right now": "turn right",
            "move 1 step right": "turn right",
            "move 1 step left": "turn left",
            "move 2 steps forward": "go forward",
            "can move 1 step right": "turn right",
            "to go forward": "go forward",
            "i need to move to the right": "turn right",
            "i need to move to the left": "turn left",
        }
        
        if self.reasoning_flag:
            # Case-insensitive search for "final answer:"
            match = re.search(r"final answer:\s*(.*?)(?:\n|$)", text_action, re.IGNORECASE)
            if match:
                extracted_action = match.group(1).strip()
                logging.debug(f"Extracted action after 'final answer:': '{extracted_action}'")
            else:
                # Fallback: Search for action-like keywords in the text
                extracted_action = text_action.lower().strip()
                potential_action = None
                
                # Check for direct variations first
                for variation, standard in action_variations.items():
                    if variation in extracted_action:
                        potential_action = standard
                        logging.debug(f"Found variation '{variation}' in text, mapping to '{standard}'")
                        break
                
                # If no variation found, look for keywords like 'right', 'left', 'forward'
                if not potential_action:
                    if "right" in extracted_action:
                        potential_action = "turn right"
                        logging.debug("Found keyword 'right' in text, mapping to 'turn right'")
                    elif "left" in extracted_action:
                        potential_action = "turn left"
                        logging.debug("Found keyword 'left' in text, mapping to 'turn left'")
                    elif "forward" in extracted_action:
                        potential_action = "go forward"
                        logging.debug("Found keyword 'forward' in text, mapping to 'go forward'")
                    elif "pick" in extracted_action:
                        potential_action = "pick up"
                        logging.debug("Found keyword 'pick' in text, mapping to 'pick up'")
                
                if potential_action:
                    extracted_action = potential_action
                    logging.debug(f"Extracted action from text: '{extracted_action}'")
                else:
                    # Split into lines and check the last line for any action
                    lines = extracted_action.split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        for variation, standard in action_variations.items():
                            if variation in line:
                                extracted_action = standard
                                logging.debug(f"Found variation '{variation}' in last line, mapping to '{standard}'")
                                break
                        if extracted_action != text_action.lower().strip():
                            break
                    if extracted_action == text_action.lower().strip():
                        logging.debug(f"No valid action found, using: '{extracted_action}'")
            
            # Apply variation mapping if needed
            extracted_action_lower = extracted_action.lower()
            for variation, standard in action_variations.items():
                if variation == extracted_action_lower:
                    extracted_action = standard
                    logging.debug(f"Mapped variation '{variation}' to '{standard}'")
                    break
            
            # Map to environment action
            action = utils.text_to_action.get(extracted_action, None)
            logging.debug(f"Mapped action: {action}")
            
            if action is None:
                self.consecutive_invalid_actions += 1
                logging.warning(
                    f"Invalid action: '{extracted_action}'. "
                    f"Consecutive invalid actions: {self.consecutive_invalid_actions}"
                )
                invalid_action_message = (
                    f"Invalid format or action. Current observation: {self.last_observation}\n"
                    "Think step-by-step (max 10 words) and end with 'final answer: [action]', where [action] is one of: "
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
                self.last_observation = text_obs
                self.consecutive_invalid_actions = 0  # Reset on valid action
                logging.debug(f"Valid action executed: {action}, Reward: {reward}, Done: {done}")
        else:
            extracted_action = text_action.strip()
            extracted_action_lower = extracted_action.lower()
            for variation, standard in action_variations.items():
                if variation == extracted_action_lower:
                    extracted_action = standard
                    logging.debug(f"Mapped non-reasoning variation '{variation}' to '{standard}'")
                    break
            action = utils.text_to_action.get(extracted_action, None)
            logging.debug(f"Non-reasoning action: '{extracted_action}', Mapped action: {action}")
            
            if action is None:
                self.consecutive_invalid_actions += 1
                logging.warning(
                    f"Invalid non-reasoning action: '{extracted_action}'. "
                    f"Consecutive invalid actions: {self.consecutive_invalid_actions}"
                )
                invalid_action_message = (
                    f"Invalid action. Current observation: {self.last_observation}\n"
                    "Valid actions are: "
                    + ", ".join(utils.text_to_action.keys())
                    + ".\n"
                    "Output one action only."
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
                self.last_observation = text_obs
                self.consecutive_invalid_actions = 0  # Reset on valid action
                logging.debug(f"Valid non-reasoning action executed: {action}, Reward: {reward}, Done: {done}")
        
        return text_obs, reward, done