import numpy as np
import torch
import torch.optim as optim

from marl.actor import Actor
from marl.critic import Critic


# ======================================================
# Replay Buffer
# ======================================================
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
            torch.FloatTensor(np.array(actions)).squeeze(1),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
        )

    def __len__(self):
        return len(self.buffer)


# ======================================================
# MADDPG Controller
# ======================================================
class MADDPG:
    def __init__(
        self,
        agent_obs_dims,
        agent_act_dims,
        global_state_dim,
        lr=1e-3,
        gamma=0.95
    ):
        self.num_agents = len(agent_obs_dims)
        self.gamma = gamma

        # ----------------------
        # Actors (one per agent)
        # ----------------------
        self.actors = [
            Actor(obs_dim, act_dim)
            for obs_dim, act_dim in zip(agent_obs_dims, agent_act_dims)
        ]

        # ----------------------
        # Centralized Critic
        # ----------------------
        self.critic = Critic(
            state_dim=global_state_dim,
            total_act_dim=sum(agent_act_dims)
        )

        # ----------------------
        # Optimizers
        # ----------------------
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr)
            for actor in self.actors
        ]

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr
        )

        # ----------------------
        # Replay Buffer
        # ----------------------
        self.replay_buffer = ReplayBuffer()

    # ==================================================
    # Action Selection
    # ==================================================
    def select_actions(self, obs_list):
        """
        obs_list: list of torch tensors (one per agent)
        returns: concatenated actions tensor
        """
        actions = []
        for actor, obs in zip(self.actors, obs_list):
            action = actor(obs)
            actions.append(action)

        return torch.cat(actions, dim=-1)

    # ==================================================
    # Reward (Simplified, Section 4 placeholder)
    # ==================================================
    def compute_reward(self, env):
        infection_penalty = env.I.sum()

        mobility_reward = sum(
            data["base_mobility"] * data["economic_weight"]
            for _, data in env.graph.nodes(data=True)
        )

        reward = -infection_penalty + 0.1 * mobility_reward
        return reward

    # ==================================================
    # Update Networks (CRITICAL PART)
    # ==================================================
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states = \
            self.replay_buffer.sample(batch_size)

        # ----------------------
        # Critic Update
        # ----------------------
        q_values = self.critic(states, actions)

        with torch.no_grad():
            target_q = rewards  # no target networks yet

        critic_loss = torch.mean((q_values - target_q) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------
        # Actor Updates
        # ----------------------
        for i, actor in enumerate(self.actors):
            # Each actor sees full global state for now
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
