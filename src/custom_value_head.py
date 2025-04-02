# custom_value_head.py
import torch.nn as nn


class CustomValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),  # New layer
            nn.ReLU(),  # Activation for the new layer
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)
