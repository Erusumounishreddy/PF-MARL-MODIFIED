class BaseAgent:
    def __init__(self, agent_id, obs_dim, act_dim):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def observe(self, env):
        """
        Extract agent-specific observation from environment.
        """
        raise NotImplementedError

    def act(self, observation):
        """
        Return an action (no learning yet).
        """
        raise NotImplementedError
