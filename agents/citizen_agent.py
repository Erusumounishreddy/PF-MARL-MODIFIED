import numpy as np
from agents.base_agent import BaseAgent

class CitizenAgent(BaseAgent):
    def observe(self, env):
        obs = []
        for _, data in env.graph.nodes(data=True):
            obs.append(data["economic_weight"])
        return np.array(obs)

    def act(self, observation):
        # Placeholder: moderate compliance
        return np.ones(len(observation)) * 0.6
