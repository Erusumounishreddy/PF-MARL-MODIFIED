import torch
from marl.actor import Actor
from marl.critic import Critic

actor = Actor(obs_dim=10, act_dim=3)
obs = torch.rand(1, 10)
print("Actor output:")
print(actor(obs))

critic = Critic(state_dim=40, total_act_dim=12)
state = torch.rand(1, 40)
actions = torch.rand(1, 12)
print("\nCritic output:")
print(critic(state, actions))
