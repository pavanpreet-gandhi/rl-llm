import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from rich.pretty import pprint
from types import SimpleNamespace
from tqdm import tqdm
import os, sys

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

args = {
        # Training config
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "env_id": "BabyAI-GoToObj-v0",
        "num_shared_layers": None,
        "num_steps_train": 2000,
        "num_envs": 4,
        "seed" : 30,
        # PPO config
        "batch_size": 4,
        "mini_batch_size": 4,
        # "gradient_accumulation_steps": 4, 
        "optimize_device_cache": True,
        "early_stopping": False,
        # Env config
        "consecutive_invalid_actions_allowed": 5,
        "invalid_action_penalty": -0.1,
        "max_steps_per_episode": 100,
        # Generation kwargs
        "max_new_tokens": 10,
        "do_sample": True,
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.90,
        # PEFT config
        "use_peft": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
    }
args = SimpleNamespace(**args)  # same type as argparse would return

envs = []
for i in range(args.num_envs):
    env = gym.make(args.env_id)
    env.seed(100 * args.seed + i)
    envs.append(env)

tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_id)

ref_model = create_reference_model(model, num_shared_layers=args.num_shared_layers)
config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_device_cache=args.optimize_device_cache,
        early_stopping=args.early_stopping
    )
trainer = PPOTrainer(config, model, ref_model, tokenizer)

generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }

system_prompt_msg = """You are an agent playing a simple navigation game. Your goal is to **{goal}**. The following are the possible actions you can take in the game, followed by a short description of each action:

turn left: turn to the left,
turn right: turn to the right,
go forward: take one step forward,
pick up: pick up the object below you,
drop: drop the object that you are holding,
toggle: manipulate the object in front of you.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.
- Never use brackets in your response!!!

PLAY!
"""

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

num_envs = args.num_envs
obss, infos = zip(*[env.reset() for env in envs])
missions = [obs["mission"] for obs in obss]
text_obss = ['\n'.join(info['descriptions']) for info in infos]
contexts = [[] for _ in range(num_envs)]
for messages, mission, text_obs in zip(contexts, missions, text_obss):
    system_prompt = system_prompt_msg.replace("{goal}", mission)
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text_obs})

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

while True:
    query_tenors_step_trial = []
    for conv in contexts:
        query_tenors_step_trial.append(tokenizer.apply_chat_template(
            conv, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).squeeze(0))
        
    generated_tokens_trial = trainer.generate(
        query_tensor=query_tenors_step_trial,
        generation_kwargs=generation_kwargs,
        return_prompt=False
    )
    response_texts = tokenizer.batch_decode(generated_tokens_trial, skip_special_tokens=True)

    for i, (env, action_text) in enumerate(zip(envs, response_texts)):
        # print('taking action: ', action_text, "in environment number: ", i, type(action_text))
        action = text_to_action.get(action_text.lower(), None)

        if action is None:
            # print('Not valid')
            text_obs = "You entered an invalid action, say nothing other than: " + str(list(text_to_action.keys()))
            reward = -0.1
            done = False
            success = False
        else:
            obs, reward, done, info = env.step(action)
            if reward > 0:
                # print(obs)
                print('reward', reward)
            text_obs = '\n'.join(info["descriptions"])
            success = True
        contexts[i].append({"role": "assistant", "content": action_text.lower()})
        contexts[i].append({"role": "user", "content": text_obs})
    