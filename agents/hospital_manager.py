import numpy as np
from agents.base_agent import BaseAgent

class HospitalManagerAgent(BaseAgent):
    def observe(self, env):
        obs = []
        for i, data in env.graph.nodes(data=True):
            is_medical = 1.0 if data["district_type"] == "medical" else 0.0
            obs.extend([
                env.I[i],
                is_medical
            ])
        return np.array(obs)

    def act(self, observation):
        # Placeholder: equal resource allocation
        num_districts = len(observation) // 2
        return np.ones(num_districts)
