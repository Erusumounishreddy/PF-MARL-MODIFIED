import numpy as np
from agents.base_agent import BaseAgent

class TransportControllerAgent(BaseAgent):
    def observe(self, env):
        obs = []
        for i, data in env.graph.nodes(data=True):
            obs.extend([
                data["base_mobility"],
                env.I[i]
            ])
        return np.array(obs)

    def act(self, observation):
        num_districts = len(observation) // 2
        return np.ones(num_districts) * 0.7
