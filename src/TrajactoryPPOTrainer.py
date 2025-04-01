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

    def compute_gae_batch(
        self, 
        rewards: List[List[float]], 
        values: List[List[float]], 
        dones: List[List[bool]]
    ) -> List[List[float]]:
        """Batch-compute GAE advantages for multiple trajectories."""
        batch_advantages = []
        for traj_rewards, traj_values, traj_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0
            next_value = 0 if traj_dones[-1] else traj_values[-1] # Bootstrap if not terminal
            
            for t in reversed(range(len(traj_rewards))):
                delta = traj_rewards[t] + self.gamma * next_value * (1 - traj_dones[t]) - traj_values[t]
                advantage = delta + self.gamma * self.gae_lambda * (1 - traj_dones[t]) * last_advantage
                advantages.append(advantage)
                last_advantage = advantage
                next_value = traj_values[t]
            advantages.reverse()  # Reverse to get the correct order
            batch_advantages.append(advantages)
        return batch_advantages

    @PPODecorators.empty_device_cache()
    def train_on_batch(self, 
                       batch_queries: List[List[torch.LongTensor]],
                       batch_responses: List[List[torch.LongTensor]], 
                       batch_rewards: List[List[float]], 
                       batch_dones: List[List[bool]]):
        """Train on a batch of trajectories."""
        
        assert len(batch_queries) == len(batch_responses) == len(batch_rewards) == len(batch_dones), "Mismatched batch sizes"
        
        batch_values = []
        
        self.model.eval()
        
        for i in range(len(batch_queries)):
            this_traj_queries = [q.to(self.current_device, non_blocking=True) for q in batch_queries[i]]
            this_traj_responses = [r.to(self.current_device, non_blocking=True) for r in batch_responses[i]]
            model_inputs = self.prepare_model_inputs(this_traj_queries, this_traj_responses)
            
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
            
            this_size = len(this_traj_queries)
            mini_batch_size = 8
            
            mini_batch_values = []
            
            for i in range(math.ceil(this_size / mini_batch_size)):
                input_kwargs = {key: value[i * mini_batch_size : (i + 1) * mini_batch_size] for key, value in model_inputs.items()}
                with torch.no_grad():
                    values = self.model(**input_kwargs)[2][:, 0].cpu().tolist()
                    mini_batch_values.extend(values)
            batch_values.append(mini_batch_values)
            del this_traj_queries, this_traj_responses, model_inputs, values, mini_batch_values
            torch.cuda.empty_cache()
        self.model.train()
        
        # Compute GAE advantages
        batch_advantages = self.compute_gae_batch(batch_rewards, batch_values, batch_dones)
        
        all_advantages = torch.cat([torch.tensor(adv) for adv in batch_advantages])
    
        # Normalize globally
        norm_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # Reshape back if needed (e.g., for logging)
        norm_advantages = norm_advantages.split([len(traj) for traj in batch_advantages])
        
        # Flatten for PPO.step()
        flat_advantages = torch.cat(norm_advantages).flatten().tolist()
        
        # Flatten the batch
        flat_batch_queries = [query for traj in batch_queries for query in traj]
        flat_batch_responses = [response for traj in batch_responses for response in traj]
        flat_advantages = [torch.tensor(a, dtype=torch.float32) for a in flat_advantages]
        
        flat_batch_queries = flat_batch_queries[:self.config.batch_size]
        flat_batch_responses = flat_batch_responses[:self.config.batch_size]
        flat_advantages = flat_advantages[:self.config.batch_size]
        
        stats = self.step(
            flat_batch_queries,
            flat_batch_responses,
            flat_advantages,  # Using GAE as reward
        )
        return stats