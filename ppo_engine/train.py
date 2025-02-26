import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from rich.pretty import pprint
from types import SimpleNamespace
from tqdm import tqdm

import gym
import babyai_text
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model

import utils
from sample_trajectory import sample_trajectory


def parse_args(logger: logging.Logger) -> Dict[str, Any]:
    """
    Parse command line arguments.
    TODO: Implement argument parsing using argparse or similar library.
    """
    args = {
        # Others
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "env_id": "BabyAI-GoToLocal-v0",
        "num_shared_layers": 6,
        "max_steps_env": 5,
        "num_steps_train": 5,
        
        # PPO config
        "batch_size": 4,
        "mini_batch_size": 4,
        
        # Generation kwargs
        "max_new_tokens": 20,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        # "repetition_penalty": 1.0,
    }
    args = SimpleNamespace(**args)
    logger.info(f"Parsed arguments: {args}")
    return args


def setup_training(args, logger: logging.Logger):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    env = gym.make(args.env_id)
    logger.info(f"Created environment: {args.env_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_id).to(device)
    logger.info("Loaded model and tokenizer")
    
    ref_model = create_reference_model(model, num_shared_layers=args.num_shared_layers)
    logger.info(f"Created reference model with {args.num_shared_layers} shared layers")
    
    config = PPOConfig(
        batch_size=args.batch_size, 
        mini_batch_size=args.mini_batch_size
    )
    trainer = PPOTrainer(config, model, ref_model, tokenizer)
    logger.info("Initialized PPO Trainer")
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }
    logger.info("Set up generation kwargs")
    
    return env, trainer, tokenizer, generation_kwargs


def train(args, logger: logging.Logger):
    
    env, trainer, tokenizer, generation_kwargs = setup_training(args, logger)
    
    logger.info("Starting training loop")
    for step in tqdm(range(args.num_steps_train)):
        
        # Collect experiences
        logger.info("Collecting experiences")
        query_tensors, response_tensors, rewards, messages = [], [], [], []
        
        while len(rewards) < args.batch_size:
            
            query_tensors_ep, response_tensors_ep, rewards_ep, messages_ep = sample_trajectory(
                env, trainer, tokenizer, generation_kwargs, args.max_steps_env
            )
            query_tensors.extend(query_tensors_ep)
            response_tensors.extend(response_tensors_ep)
            rewards.extend(rewards_ep)
            messages.extend(messages_ep)
            
            logger.info(f"Collected {len(rewards)} experiences")
            logger.info(f"Messages: {messages}")
        
        query_tensors = query_tensors[:args.batch_size]
        response_tensors = response_tensors[:args.batch_size]
        rewards = rewards[:args.batch_size]
        
        # Train
        break
        

if __name__ == "__main__":
    logger = utils.create_logger("train")
    args = parse_args(logger)
    train(args, logger)
    