from envs.urban_env import UrbanEpidemicEnv
import numpy as np

env = UrbanEpidemicEnv(num_districts=6)

obs, _ = env.reset()

print("Initial infection levels:")
print(env.I)

for day in range(10):
    action = np.zeros(env.num_districts)  # no policy intervention
    obs, _, _, _, _ = env.step(action)
    print(f"Day {day+1} infection:", env.I)
