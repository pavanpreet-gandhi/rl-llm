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
from peft import LoraConfig, get_peft_model
import wandb

import utils
from env_manager import EnvManager
from sample_trajectories import sample_trajectories


def parse_args() -> Dict[str, Any]:
    """
    Parse command training configuration arguments.
    TODO: Other hyperparameters (e.g. learning_rate, ppo_epochs, kl stuff, cliprange, vf_coeff, whiten_rewards, etc.)
    TODO: Choose generation kwargs
    """
    args = {
        # Others
        "project_name": "babyai-ppo",
        "experiment_name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "env_id": "BabyAI-GoToLocal-v0",
        "num_shared_layers": 6,
        "num_steps_train": 5,
        "num_envs": 4,
        
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

        # PEFT config
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
    }
    args = SimpleNamespace(**args) # same type as argparse would return
    return args


def setup_training(args, logger: logging.Logger):
    """
    Set up everything required for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    env_managers = [EnvManager(gym.make("BabyAI-MixedTrainLocal-v0", seed=i)) for i in range(args.num_envs)]
    logger.info(f"Created environment: {args.env_id}")

    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )
        logger.info(f"Using PEFT")
    else:
        peft_config = None
        logger.info("Not using PEFT")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_id, peft_config=peft_config).to(device)
    logger.info("Loaded model and tokenizer")
    
    ref_model = create_reference_model(model, num_shared_layers=args.num_shared_layers)
    logger.info(f"Created reference model with {args.num_shared_layers} shared layers")
    
    config = PPOConfig(
        batch_size=args.batch_size, 
        mini_batch_size=args.mini_batch_size,
        is_peft_model=args.use_peft,
        exp_name=args.experiment_name,
        log_with="wandb",
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
    
    return env_managers, trainer, tokenizer, generation_kwargs, device


def train(args, logger: logging.Logger):
    """
    Main training loop.
        1. Collect experiences
        2. Train PPO
        3. Log stats
        4. Repeat
    """
    env_managers, trainer, tokenizer, generation_kwargs, device = setup_training(args, logger)
    
    logger.info("STARTING TRAINING LOOP")
    for step in tqdm(range(args.num_steps_train)):
        
        # Collect experiences
        logger.info("COLLECTING EXPERIENCES...")
        query_tensors, response_tensors, rewards = sample_trajectories(
            env_managers,
            trainer,
            tokenizer,
            generation_kwargs,
            device,
            experiences_needed=args.batch_size,
            logger=logger,
        )
        query_tensors = query_tensors[:args.batch_size]
        response_tensors = response_tensors[:args.batch_size]
        rewards = rewards[:args.batch_size]
        
        # Train step
        stats = trainer.step(query_tensors, response_tensors, rewards)

        # Log stats
        query = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        response = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch = {'query': query, 'response': response}
        trainer.log_stats(stats, batch, rewards)
        logger.info(f"TRAINING STEP {step} COMPLETED")


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project=args.project_name, name=args.experiment_name)
    logger = utils.create_logger(args.experiment_name, console_output=True)
    logger.info(f"Using arguments: {args}")
    train(args, logger)