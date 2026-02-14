import numpy as np
from agents.base_agent import BaseAgent

class PolicyMakerAgent(BaseAgent):
    def observe(self, env):
        obs = []
        for i, data in env.graph.nodes(data=True):
            obs.extend([
                env.I[i],                  # infection level
                data["economic_weight"]    # economic importance
            ])
        return np.array(obs)

    def act(self, observation):
        # Placeholder: neutral policy
        num_districts = len(observation) // 2
        return np.ones(num_districts) * 0.5
