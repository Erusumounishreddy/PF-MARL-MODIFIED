from envs.urban_env import UrbanEpidemicEnv
import numpy as np

env = UrbanEpidemicEnv(num_districts=5)

obs, _ = env.reset()
print("Initial observation (per district):")
print(obs.reshape(5, 3))

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("\nAfter one step (per district):")
print(obs.reshape(5, 3))
