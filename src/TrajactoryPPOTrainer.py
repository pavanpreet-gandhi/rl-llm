import torch
from typing import List
from trl import PPOTrainer, PPOConfig
from trl.core import PPODecorators
import math

class BatchedTrajectoryPPOTrainer(PPOTrainer):
    def __init__(self, config: PPOConfig, model, ref_model, tokenizer, gamma=0.9, gae_lambda=0.95):
        super().__init__(config, model, ref_model, tokenizer)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    @PPODecorators.empty_device_cache()
    def compute_gae(self, 
                    queries: List[torch.LongTensor],
                    responses: List[torch.LongTensor], 
                    rewards: List[float], 
                    dones: List[bool]):
        
        assert len(queries) == len(responses) == len(rewards) == len(dones), "Mismatched trajactory sizes"
        
        traj_values = []
        
        self.model.eval()
        
        traj_queries = [q.to(self.current_device, non_blocking=True) for q in queries]
        traj_responses = [r.to(self.current_device, non_blocking=True) for r in responses]
        model_inputs = self.prepare_model_inputs(traj_queries, traj_responses)
        
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )
            
        this_size = len(traj_queries)
        mini_batch_size = 8
        
        for i in range(math.ceil(this_size / mini_batch_size)):
            input_kwargs = {key: value[i * mini_batch_size : (i + 1) * mini_batch_size] for key, value in model_inputs.items()}
            with torch.no_grad():
                values = self.model(**input_kwargs)[2][:, 0].cpu().tolist()
                traj_values.extend(values)
        del traj_queries, traj_responses, model_inputs, values
        torch.cuda.empty_cache()
        self.model.train()
        
        # Compute GAE advantages
        advantages = []
        last_advantage = 0
        next_value = 0 if dones[-1] else traj_values[-1] # Bootstrap if not terminal
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - traj_values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages.append(advantage)
            last_advantage = advantage
            next_value = traj_values[t]
        advantages.reverse()  # Reverse to get the correct order
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
    
        # Normalize globally
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        norm_advantages = norm_advantages.tolist()
        
        return norm_advantages