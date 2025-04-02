import torch.nn as nn

class CustomValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Final layer exposed directly for TRL compatibility
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        x = self.mlp(hidden_states)
        return self.summary(x)
