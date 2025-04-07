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
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from peft import LoraConfig, get_peft_model
import wandb
from huggingface_hub import HfApi, create_repo, hf_hub_download

import utils
from env_manager import EnvManager
from sample_batch import sample_batch, EpisodeCounter
from custom_value_head import CustomValueHead
from TrajactoryPPOTrainer import BatchedTrajectoryPPOTrainer, log_memory


def parse_args() -> Dict[str, Any]:
    """
    Parse command training configuration arguments.
    """
    args = {
        # Logging config
        "project_name": "babyai-classical-ppo-prefinal-experiments",  # TODO: "babyai-ppo-experiments"
        "experiment_name": "mix_5_no_50_0.9_0.7", #"mix_5_no_reason_50_0.9_0.7",
        "entity": "OE_2025",
        "push_to_hub": True, # TODO: True
        "hub_model_id": None, # If None, will use f"{hf_username}/{args.project_name}-{args.experiment_name}"
        # Checkpoint config
        "save_every": 10,  # TODO: 10
        "checkpoint_dir": "checkpoints",
        # Load pretrained model
        "pretrained_dir": "CatkinChen/babyai-classical-ppo-experiments-2025-04-03_13-12-13",  # add path for the pretrained model "your-hf-username/your-model-repo"
        "load_checkpoint": None,
        # Training config
        "model_id": "meta-llama/Llama-3.2-3B-Instruct", # "HuggingFaceTB/SmolLM2-135M-Instruct", ,
        "separate_vhead": False, 
        "num_shared_layers": None,
        "num_steps_train": 500,
        "num_envs": 4,  # TODO: 4
        # PPO config
        "batch_size": 128,  # TODO: 128
        "mini_batch_size": 16,  # TODO: 64
        "optimize_device_cache": False,
        "early_stopping": False,
        "learning_rate": 1.41e-5,
        "kl_penalty" : "kl", # default "kl"
        # Env config
        "env_ids": ["BabyAI-GoTo-v0", "BabyAI-Pickup-v0"],
        "consecutive_invalid_actions_allowed": 5,
        "invalid_action_penalty": -2,
        "context_window": 5,  # Number of previous experiences to keep in context
        "reasoning_flag": True,
        # Generation kwargs
        "min_length": -1,  # don't ignore the EOS token
        "top_k": 50,  # no top-k sampling
        "top_p": 0.9,  # no nucleus sampling
        "do_sample": True,  # yes, we want to sample
        "max_new_tokens": 15,
        "temperature": 0.7,
        # PEFT config
        "use_peft": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        # RL config
        "trajactory_rl": True,
        "gamma": 0.9,
        "lam": 0.95,
    }
    args = SimpleNamespace(**args)  # same type as argparse would return
    return args


def setup_training(args, logger: logging.Logger):
    """
    Set up everything required for training.
    """
    # Set up device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set up environment managers
    envs = [
        EnvManager(
            env_ids=args.env_ids,
            invalid_action_penalty=args.invalid_action_penalty,
            consecutive_invalid_actions_allowed=args.consecutive_invalid_actions_allowed,
            reasoning_flag=args.reasoning_flag,
        )
        for i in range(args.num_envs)
    ]
    logger.info(f"Created environments: {args.env_ids}")

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
    if args.load_checkpoint:
        # Load model and tokenizer from checkpoint
        pretrained_dir = args.pretrained_dir
        # Load the base model and tokenizer
        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_dir, torch_dtype=torch.bfloat16)
        if args.separate_vhead:
            hidden_size = model.config.hidden_size
            model.v_head = CustomValueHead(hidden_size)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

        # Load the value head weights
        value_head_path = hf_hub_download(
            repo_id=pretrained_dir, filename="value_head.bin"
        )
        model.v_head.load_state_dict(
            torch.load(value_head_path)
        )  # .to(device).to(dtype=torch.bfloat16)

        logger.info(f"Loaded model and tokenizer from {pretrained_dir}")

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_id, peft_config=peft_config, torch_dtype=torch.bfloat16
        ).to(device)
        if args.separate_vhead:
            hidden_size = model.config.hidden_size
            model.v_head = (
                CustomValueHead(hidden_size).to(device).to(dtype=torch.bfloat16)
            )
        logger.info("Loaded model and tokenizer from scratch")

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
        kl_penalty=args.kl_penalty
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
        "pad_token_id": tokenizer.pad_token_id,
    }
    logger.info("Set up generation kwargs")

    # Set up HuggingFace Hub repository if needed
    if args.push_to_hub:
        if args.hub_model_id is None:
            hf_username = HfApi().whoami()["name"]
            args.hub_model_id = (
                f"{hf_username}/{args.project_name}-{args.experiment_name}"
            )
        try:
            create_repo(args.hub_model_id, exist_ok=True)
            logger.info(f"Created HuggingFace Hub repo: {args.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            logger.info(f"Continuing without pushing to hub")
            args.push_to_hub = False

    return envs, trainer, tokenizer, generation_kwargs, device, checkpoint_dir


def train(args, logger: logging.Logger):
    """
    Main training loop.
    """
    # Set up training
    env_managers, trainer, tokenizer, generation_kwargs, device, checkpoint_dir = (
        setup_training(args, logger)
    )
    # Log key arguments to wandb
    wandb.config.update(
        {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning_flag": args.reasoning_flag,
            "batch_size": args.batch_size,
            "mini_batch_size": args.mini_batch_size,
            "context_window": args.context_window,
            "max_new_tokens": args.max_new_tokens,
            "model_id": args.model_id,
            "separate_vhead": args.separate_vhead,
            "env_ids": args.env_ids,
            "num_envs": args.num_envs,
        }
    )
    logger.info("Logged key arguments to wandb")

    episode_counter = EpisodeCounter()
    logger.info("STARTING TRAINING LOOP")

    # Create logger for training
    train_logger = logging.getLogger("train_logger")
    train_logger.setLevel(logging.INFO)
    # Remove any existing handlers to avoid duplicate logging
    # Create a file handler to write logs to a file
    train_log_file = "logs/training_logs.txt"
    file_handler = logging.FileHandler(train_log_file)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    train_logger.handlers = []  # Clear existing handlers
    # Add only the file handler
    train_logger.addHandler(file_handler)
    # Ensure logs are not printed to the console
    train_logger.propagate = False

    # Add the handler to the logger
    train_logger.addHandler(file_handler)
    wandb.save(train_log_file)

    for step in tqdm(range(args.num_steps_train)):
        train_logger.info(f"TRAINING STEP {step + 1} STARTED")

        # Collect experiences
        logger.info("COLLECTING EXPERIENCES...")
        start_time = datetime.now()
        queries, responses, rewards, stats = sample_batch(
            envs=env_managers,
            tokenizer=tokenizer,
            trainer=trainer,
            generation_kwargs=generation_kwargs,
            device=device,
            batch_size=args.batch_size,
            context_window=args.context_window,
            reasoning_flag=args.reasoning_flag,
            logger=train_logger,
            trajectory_rl=args.trajactory_rl,
            episode_counter=episode_counter
        )
        sample_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Sample batch time: {sample_time:.2f} seconds")

        # Log sampling stats to wandb
        stats["sample_time"] = sample_time
        stats["sampled_batch_size"] = len(rewards)

        # Select random subset of experiences (since sample_trajectories could return more than needed)
        indices = torch.randperm(len(rewards))[: args.batch_size].tolist()
        queries = [queries[i] for i in indices]
        responses = [responses[i] for i in indices]
        rewards = [rewards[i] for i in indices]

        # Train step
        log_memory(logger, "Before trainer step")
        start_time = datetime.now()
        trainer_stats = trainer.step(queries, responses, rewards)
        train_time = (datetime.now() - start_time).total_seconds()
        log_memory(logger, "After trainer step")
        logger.info(f"Trainer step time: {train_time:.2f} seconds")

        # Add timing stats to wandb
        stats.update(trainer_stats)
        stats["train_step_time"] = train_time
        stats["total_time"] = sample_time + train_time
        logger.info(f"TRAINING STEP {step} COMPLETED")

        # Log trainer stats
        query = tokenizer.batch_decode(queries, skip_special_tokens=True)
        response = tokenizer.batch_decode(responses, skip_special_tokens=True)
        batch = {"query": query, "response": response}
        trainer.log_stats(stats, batch, rewards)

        # Upload the training log file to wandb
        train_logger.info("Uploaded training_logs.txt to wandb")
        wandb.save("training_logs.txt")

        # Save checkpoint if needed
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{step + 1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            trainer.model.save_pretrained(
                checkpoint_path
            )  # saves either the full model or just the PEFT adapters
            # Save the value head weights
            value_head_path = os.path.join(checkpoint_path, "value_head.bin")
            torch.save(trainer.model.v_head.state_dict(), value_head_path)

            tokenizer.save_pretrained(checkpoint_path)
            logger.info(
                f"Saved model checkpoint at step {step + 1} to {checkpoint_path}"
            )

            # Push to HuggingFace Hub if needed
            if args.push_to_hub:
                try:
                    api = HfApi()
                    api.upload_folder(
                        folder_path=checkpoint_path,
                        path_in_repo=".",
                        repo_id=args.hub_model_id,
                        commit_message=f"Checkpoint {step + 1}",
                        repo_type="model",
                    )
                except Exception as e:
                    logger.error(f"Failed to push to hub: {e}")
                    logger.info("Continuing with training")



if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # set up logger and wandb
    wandb.init(project=args.project_name, name=args.experiment_name, entity=args.entity)
    logger = utils.create_logger(args.experiment_name, console_output=True)
    logger.info(f"Using arguments: {args}")

    # Call the training function
    try:
        train(args, logger)
    except Exception as e:
        import traceback

        logger.error(f"Training failed: {e}\n{traceback.format_exc()}")
