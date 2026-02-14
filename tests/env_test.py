from envs.urban_env import UrbanEpidemicEnv

env = UrbanEpidemicEnv(num_districts=5)

obs, _ = env.reset()
print("Initial observation (per district):")

num_districts = env.num_districts
features_per_district = obs.shape[0] // num_districts
print(obs.reshape(num_districts, features_per_district))

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("\nAfter one step (per district):")
print(obs.reshape(num_districts, features_per_district))

