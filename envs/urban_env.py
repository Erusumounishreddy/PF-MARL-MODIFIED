import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np


class UrbanEpidemicEnv(gym.Env):
    """
    Real-world aligned synthetic urban environment for PF-MARL.
    Represents a city as interconnected administrative districts.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_districts=12, seed=42):
        super().__init__()

        self.num_districts = num_districts
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build realistic city graph
        self.graph = self._build_city_graph()

        # Observation: district-level public information
        self.features_per_node = 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_districts * self.features_per_node,),
            dtype=np.float32
        )

        # Placeholder policy intensity signal
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_districts,),
            dtype=np.float32
        )

        self.state = None

    def _build_city_graph(self):
        G = nx.Graph()

        district_types = ["residential", "commercial", "medical"]

        for i in range(self.num_districts):
            d_type = self.rng.choice(
                district_types,
                p=[0.6, 0.3, 0.1]  # realistic city composition
            )

            G.add_node(
                i,
                district_type=d_type,
                population=self.rng.uniform(0.4, 1.0),
                base_mobility=self.rng.uniform(0.3, 0.9),
                economic_weight=(
                    1.0 if d_type == "commercial"
                    else 0.6 if d_type == "medical"
                    else 0.4
                ),
            )

        # Ensure city-wide connectivity
        for i in range(self.num_districts - 1):
            G.add_edge(
                i,
                i + 1,
                mobility_weight=self.rng.uniform(0.4, 1.0)
            )

        # Add realistic cross-links
        extra_edges = self.num_districts // 2
        for _ in range(extra_edges):
            a, b = self.rng.choice(self.num_districts, size=2, replace=False)
            G.add_edge(
                a,
                b,
                mobility_weight=self.rng.uniform(0.2, 0.8)
            )

        return G

    def _get_observation(self):
        obs = []
        for _, data in self.graph.nodes(data=True):
            obs.extend([
                data["population"],
                data["base_mobility"],
                data["economic_weight"],
            ])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = self._get_observation()
        return self.state, {}

    def step(self, action):
        """
        Policy signal applied with inertia.
        No epidemiology yet — structure only.
        """
        action = np.clip(action, 0.0, 1.0)

        for i, a in enumerate(action):
            node = self.graph.nodes[i]

            # Policy inertia: no instant drastic changes
            node["base_mobility"] = np.clip(
                0.9 * node["base_mobility"] + 0.1 * (1 - a),
                0.1,
                1.0
            )

        self.state = self._get_observation()

        reward = 0.0  # Defined in later sections
        terminated = False
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self):
        pass
