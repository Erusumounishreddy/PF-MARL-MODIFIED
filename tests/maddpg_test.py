from marl.maddpg import MADDPG

# Dummy dimensions for sanity check
agent_obs_dims = [10, 10, 10, 10]
agent_act_dims = [3, 3, 3, 3]
global_state_dim = 40

maddpg = MADDPG(
    agent_obs_dims=agent_obs_dims,
    agent_act_dims=agent_act_dims,
    global_state_dim=global_state_dim
)

print("MADDPG controller initialized successfully.")
print("Number of agents:", maddpg.num_agents)
print("Number of actors:", len(maddpg.actors))
print("Replay buffer size:", len(maddpg.replay_buffer))
