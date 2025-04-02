import torch
from typing import List
from trl import PPOTrainer, PPOConfig
from trl.core import PPODecorators
import math
import logging

def log_memory(logger, point):
    logger.info(f"{point} - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                f"Max: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

class BatchedTrajectoryPPOTrainer(PPOTrainer):
    def __init__(self, config: PPOConfig, model, ref_model, tokenizer, gamma=0.9, gae_lambda=0.95):
        super().__init__(config, model, ref_model, tokenizer)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    @PPODecorators.empty_device_cache()
    def compute_returns(self, 
                    queries: List[torch.LongTensor],
                    responses: List[torch.LongTensor], 
                    rewards: List[float], 
                    dones: List[bool],
                    logger: logging.Logger = None):
        
        assert len(queries) == len(responses) == len(rewards) == len(dones), "Mismatched trajactory sizes"
        
        traj_values = []
        
        self.model.eval()
        
        traj_queries = [q.to(self.current_device, non_blocking=True) for q in queries]
        traj_responses = [r.to(self.current_device, non_blocking=True) for r in responses]
        log_memory(logger, "Before model inputs")
        model_inputs = self.prepare_model_inputs(traj_queries, traj_responses)
        log_memory(logger, "After model inputs")
        this_size = len(traj_queries)
        mini_batch_size = 8
        log_memory(logger, "Before value calculation")
        for i in range(math.ceil(this_size / mini_batch_size)):
            input_kwargs = {key: value[i * mini_batch_size : (i + 1) * mini_batch_size] for key, value in model_inputs.items()}
            with torch.no_grad():
                values = self.model(**input_kwargs)[2][:, -1].cpu().tolist()
                traj_values.extend(values)
        del traj_queries, traj_responses, model_inputs, values
        torch.cuda.empty_cache()
        self.model.train()
        log_memory(logger, "After value calculation")
        
        # Compute returns
        Gs = []
        last_G = 0
        next_value = 0 if dones[-1] else traj_values[-1] # Bootstrap if not terminal
        
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * (1.0 - self.gae_lambda) * next_value + self.gamma * self.gae_lambda * last_G
            Gs.append(G)
            last_G = G
            next_value = traj_values[t]
        Gs.reverse()  # Reverse to get the correct order
        
        return Gs