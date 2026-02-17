import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from marl.actor import Actor
from marl.critic import Critic

from opacus.accountants import RDPAccountant


# ---------------- Replay Buffer ---------------- #
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, actions, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, actions, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(
            *[self.buffer[i] for i in batch]
        )

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------- MADDPG ---------------- #
class MADDPG:
    def __init__(
        self,
        agent_obs_dims,
        agent_act_dims,
        global_state_dim,
        lr=1e-3,
        gamma=0.95,
        noise_multiplier=1.0,
        max_grad_norm=1.0
    ):
        self.num_agents = len(agent_obs_dims)
        self.gamma = gamma

        # ----- Actors ----- #
        self.actors = [
            Actor(obs_dim, act_dim)
            for obs_dim, act_dim in zip(agent_obs_dims, agent_act_dims)
        ]

        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr)
            for actor in self.actors
        ]

        # ----- Centralized Critic ----- #
        self.critic = Critic(
            state_dim=global_state_dim,
            total_act_dim=sum(agent_act_dims)
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr
        )

        # ----- Differential Privacy Accounting ----- #
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.accountant = RDPAccountant()

        self.replay_buffer = ReplayBuffer()

    # ---------- Federated helpers ---------- #
    def get_critic_weights(self):
        return self.critic.state_dict()

    def set_critic_weights(self, weights):
        self.critic.load_state_dict(weights)

    # ---------- Action selection ---------- #
    def select_actions(self, obs_list):
        actions = []
        for actor, obs in zip(self.actors, obs_list):
            action = actor(obs)
            actions.append(action)
        return torch.cat(actions, dim=-1)

    # ---------- Training update ---------- #
    def update(self, batch_size=64):

        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states = \
            self.replay_buffer.sample(batch_size)

        # ----- Critic update ----- #
        q_values = self.critic(states, actions)

        with torch.no_grad():
            target_q = rewards

        critic_loss = torch.mean((q_values - target_q) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # 🔐 DP: Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm
        )

        # 🔐 DP: Gaussian noise injection
        for p in self.critic.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=p.grad.shape
                )
                p.grad += noise

        self.critic_optimizer.step()

        # 🔐 Privacy accounting
        self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=batch_size / len(self.replay_buffer)
        )

        epsilon, _ = self.accountant.get_privacy_spent(delta=1e-5)
        print(f"[DP] Current privacy epsilon: {epsilon:.4f}")

        # ----- Actor updates ----- #
        for i, actor in enumerate(self.actors):
            obs = states

            action = actor(obs)

            all_actions = actions.clone()
            act_start = sum(
                a.fc3.out_features for a in self.actors[:i]
            )
            act_end = act_start + actor.fc3.out_features
            all_actions[:, act_start:act_end] = action

            actor_loss = -self.critic(states, all_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
