import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Centralized critic.
    Input: global state + all agent actions
    Output: Q-value
    """

    def __init__(self, state_dim, total_act_dim, hidden_dim=256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + total_act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, actions):
        """
        state: (batch, state_dim)
        actions: (batch, total_act_dim)
        """
        x = torch.cat([state, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
