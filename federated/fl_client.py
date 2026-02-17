import flwr as fl
import torch
import numpy as np

from marl.maddpg import MADDPG
from envs.urban_env import UrbanEpidemicEnv


class MADDPGClient(fl.client.NumPyClient):
    def __init__(self):
        # Initialize environment
        self.env = UrbanEpidemicEnv()

        # Get dimensions directly from Gym spaces
        obs_dim = self.env.observation_space.shape[0]
        total_act_dim = self.env.action_space.shape[0]

        num_agents = 4
        per_agent_act_dim = total_act_dim // num_agents

        # Initialize MADDPG
        self.maddpg = MADDPG(
            agent_obs_dims=[obs_dim] * num_agents,
            agent_act_dims=[per_agent_act_dim] * num_agents,
            global_state_dim=obs_dim
        )

    # -------- Federated parameter functions -------- #

    def get_parameters(self, config):
        weights = self.maddpg.get_critic_weights()
        return [v.cpu().numpy() for v in weights.values()]

    def set_parameters(self, parameters):
        keys = list(self.maddpg.get_critic_weights().keys())
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(keys, parameters)
        }
        self.maddpg.set_critic_weights(state_dict)

    # -------- Local training -------- #

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Run a few local episodes
        for _ in range(2):
            obs, _ = self.env.reset()
            done = False

            while not done:
                obs_list = [
                    torch.tensor(obs).float().unsqueeze(0)
                ] * 4

                actions = self.maddpg.select_actions(obs_list)

                next_obs, reward, terminated, truncated, _ = \
                    self.env.step(actions.detach().numpy())

                done = terminated or truncated

                self.maddpg.replay_buffer.push(
                    obs,
                    actions.detach().numpy(),
                    reward,
                    next_obs
                )

                self.maddpg.update()

                obs = next_obs

        return self.get_parameters({}), len(self.maddpg.replay_buffer), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.maddpg.replay_buffer), {}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8081",
        client=MADDPGClient()
    )
