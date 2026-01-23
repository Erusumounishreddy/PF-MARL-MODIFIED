import matplotlib.pyplot as plt
import networkx as nx
from envs.urban_env import UrbanEpidemicEnv

env = UrbanEpidemicEnv(num_districts=10)
G = env.graph

# Color by district type
colors = []
for _, data in G.nodes(data=True):
    if data["district_type"] == "residential":
        colors.append("lightblue")
    elif data["district_type"] == "commercial":
        colors.append("orange")
    else:
        colors.append("red")  # medical

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    node_color=colors,
    with_labels=True,
    node_size=800,
    font_size=8
)

plt.title("Synthetic Urban City Graph")
plt.show()
