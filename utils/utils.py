import logging
import os
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from collections import OrderedDict


def create_logger(name: str, log_dir: str = "logs", console_output: bool = False) -> logging.Logger:
    """
    Create and configure a logger with file and optionally rich console handlers.
    
    Args:
        name (str): Name of the logger
        log_dir (str): Directory to store log files
        console_output (bool): Whether to output logs to console (default: False)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}.log')
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    # Only add rich console handler if console_output is True
    if console_output:
        console = Console(theme=Theme({
            "logging.level.info": "cyan",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
        }))
        
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=True
        )
        rich_handler.setLevel(logging.INFO)
        logger.addHandler(rich_handler)
    
    return logger


def get_system_prompt() -> str:
    """
    Load the system prompt from a text file.
    
    Returns:
        str: The system prompt
    """
    with open("resources/system_prompt.txt", "r") as f:
        system_prompt = f.read()
    return system_prompt


action_to_text = {
    0: 'turn left',
    1: 'turn right',
    2: 'go forward',
    3: 'pick up',
    4: 'drop',
    5: 'toggle',
    6: 'done',
}
text_to_action = {v: k for k, v in action_to_text.items()}


task_types = ["goto", "pickup", "putnext", "open", "pick up seq go to"]


keyword_to_task = OrderedDict([
    ("then", "pick up seq go to"),
    ("after", "pick up seq go to"),
    ("go to", "goto"),
    ("pick up", "pickup"),
    ("open", "open"),
    ("put", "putnext"),
])

def get_task_from_mission(mission: str) -> str:
    """
    Extracts the task from the mission string.
    """
    for keyword, task in keyword_to_task.items():
        if keyword in mission:
            return task
    return "unknown"