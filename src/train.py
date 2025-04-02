import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from rich.pretty import pprint
from types import SimpleNamespace
from tqdm import tqdm
import os, sys
import math

import gym
import babyai_text
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from peft import LoraConfig, get_peft_model
import wandb
from huggingface_hub import HfApi, create_repo

import utils
from env_manager import EnvManager
from sample_batch import sample_batch
from TrajactoryPPOTrainer import BatchedTrajectoryPPOTrainer


def parse_args() -> Dict[str, Any]:
    """
    Parse command training configuration arguments.
    """
    args = {
        # Logging config
        "project_name": "babyai-ppo",
        "experiment_name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "push_to_hub": True,
        "hub_model_id": None, # If None, will use f"{hf_username}/{args.project_name}-{args.experiment_name}"

        # Checkpoint config
        "save_every": 50,
        "checkpoint_dir": "checkpoints",

        # Training config
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "env_id": "BabyAI-GoToObj-v0",
        "num_shared_layers": None,
        "num_steps_train": 1000,
        "num_envs": 4, # TODO: change to 8
        
        # PPO config
        "batch_size": 64, # TODO: change to 128
        "mini_batch_size": 16, # TODO: change according to memory constraints
        "optimize_device_cache": False,
        "early_stopping": False,
        "learning_rate": 1.41e-5,

        # Env config
        "consecutive_invalid_actions_allowed": 5,
        "invalid_action_penalty": -2,
        "context_window": 1, # Number of previous experiences to keep in context
        
        # Generation kwargs
        "min_length": -1, # don't ignore the EOS token
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "max_new_tokens": 10,
        "temperature": 0.8,

        # PEFT config
        "use_peft": True,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        
        # RL config
        "gamma": 0.9,
        "lam": 0.95,
    }
    args = SimpleNamespace(**args) # same type as argparse would return
    return args


def setup_training(args, logger: logging.Logger):
    """
    Set up everything required for training.
    """
    # Set up device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set up environment managers
    env_managers = [
        EnvManager(
            gym.make(args.env_id, seed=i), 
            invalid_action_penalty=args.invalid_action_penalty,
            consecutive_invalid_actions_allowed=args.consecutive_invalid_actions_allowed,
        )
        for i in range(args.num_envs)
    ]
    logger.info(f"Created environment: {args.env_id}")

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory created at {checkpoint_dir}")

    # Create PEFT config if using PEFT
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
    
    # Create model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_id, 
        peft_config=peft_config, 
        load_in_4bit=True
    ).to(device)
    logger.info("Loaded model and tokenizer")
    
    # Create reference model
    ref_model = create_reference_model(model, num_shared_layers=args.num_shared_layers)
    logger.info(f"Created reference model with {args.num_shared_layers} shared layers")
    
    # Set up PPOTrainer object
    config = PPOConfig(
        batch_size=args.batch_size, 
        mini_batch_size=args.mini_batch_size,
        optimize_device_cache=args.optimize_device_cache,
        early_stopping=args.early_stopping,
        is_peft_model=args.use_peft,
        exp_name=args.experiment_name,
        log_with="wandb",
    )
    trainer = BatchedTrajectoryPPOTrainer(config, model, ref_model, tokenizer, args.gamma, args.lam)
    logger.info("Initialized PPO Trainer")
    
    # Set up generation kwargs for sampling trajectories
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }
    logger.info("Set up generation kwargs")

    # Set up HuggingFace Hub repository if needed
    if args.push_to_hub:
        if args.hub_model_id is None:
            hf_username = HfApi().whoami()["name"]
            args.hub_model_id = f"{hf_username}/{args.project_name}-{args.experiment_name}"
        try:
            create_repo(args.hub_model_id, exist_ok=True)
            logger.info(f"Created HuggingFace Hub repo: {args.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            logger.info(f"Continuing without pushing to hub")
            args.push_to_hub = False
    
    return env_managers, trainer, tokenizer, generation_kwargs, device, checkpoint_dir


def train(args, logger: logging.Logger):
    """
    Main training loop.
    """
    # Set up training
    env_managers, trainer, tokenizer, generation_kwargs, device, checkpoint_dir = setup_training(args, logger)
    
    logger.info("STARTING TRAINING LOOP")
    for step in tqdm(range(args.num_steps_train)):
        
        # Collect experiences
        logger.info("COLLECTING EXPERIENCES...")
        start_time = datetime.now()
        query_tensors, response_tensors, rewards, sampling_stats = sample_batch(
            env_managers,
            tokenizer,
            trainer,
            generation_kwargs,
            batch_size=args.batch_size,
            logger=logger,
            context_window=args.context_window,
        )
        sample_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Sample batch time: {sample_time:.2f} seconds")
        
        # Select random subset of experiences (since sample_trajectories could return more than needed)
        indices = torch.randperm(len(rewards))[:args.batch_size].tolist()
        query_tensors = [query_tensors[i] for i in indices]
        response_tensors = [response_tensors[i] for i in indices]
        rewards = [rewards[i] for i in indices]
        # Log sampling stats to wandb
        wandb.log({
            "success_rate": sampling_stats["success_rate"],
            "total_count": sampling_stats["total_count"],
            "success_count": sampling_stats["success_count"],
            "avg_success_reward": sampling_stats["avg_success_reward"],
            "sample_batch_time": sample_time
        })
        
        # Train step
        start_time = datetime.now()
        stats = trainer.step(query_tensors, response_tensors, rewards)
        train_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Trainer step time: {train_time:.2f} seconds")

        # Log stats
        query = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        response = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch = {'query': query, 'response': response}
        trainer.log_stats(stats, batch, rewards)
        # Add timing stats to wandb
        wandb.log({"trainer_step_time": train_time})
        logger.info(f"TRAINING STEP {step} COMPLETED")
        
        # Save checkpoint if needed
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{step + 1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            trainer.model.save_pretrained(checkpoint_path) # saves either the full model or just the PEFT adapters
            logger.info(f"Saved model checkpoint at step {step + 1} to {checkpoint_path}")
        
            # Push to HuggingFace Hub if needed
            if args.push_to_hub:
                try:
                    trainer.model.push_to_hub(
                        args.hub_model_id, 
                        commit_message=f"Training checkpoint {step + 1}",
                    )
                except Exception as e:
                    logger.error(f"Failed to push to hub: {e}")
                    logger.info("Continuing with training")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # set up logger and wandb
    wandb.init(project=args.project_name, name=args.experiment_name)
    logger = utils.create_logger(args.experiment_name, console_output=True)
    logger.info(f"Using arguments: {args}")

    # Call the training function
    try:
        train(args, logger)
    except Exception as e:
        import traceback
        logger.error(f"Training failed: {e}\n{traceback.format_exc()}")