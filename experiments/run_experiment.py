def run_training_episode(env, maddpg):
    state, _ = env.reset()
    done = False

    while not done:
        obs_list = [
            torch.FloatTensor(state).unsqueeze(0)
            for _ in range(maddpg.num_agents)
        ]

        actions = maddpg.select_actions(obs_list)
        next_state, reward, terminated, truncated, _ = env.step(
            actions.detach().numpy()
        )

        maddpg.replay_buffer.push(
            state, actions.detach().numpy(), reward, next_state
        )

        maddpg.update(batch_size=32)

        state = next_state
        done = terminated or truncated

import torch
from envs.urban_env import UrbanEpidemicEnv
from marl.maddpg import MADDPG

env = UrbanEpidemicEnv(num_districts=6)

agent_obs_dims = [env.features_per_node * env.num_districts] * 4
agent_act_dims = [env.num_districts] * 4
global_state_dim = env.features_per_node * env.num_districts

maddpg = MADDPG(
    agent_obs_dims=agent_obs_dims,
    agent_act_dims=agent_act_dims,
    global_state_dim=global_state_dim
)

episodes = 5
steps_per_episode = 20

for ep in range(episodes):
    state, _ = env.reset()

    for step in range(steps_per_episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        obs_list = [state_tensor] * 4
        actions = maddpg.select_actions(obs_list)

        # actions shape: (1, 4*num_districts)
        actions_np = actions.detach().numpy().reshape(4, env.num_districts)

# Aggregate agent decisions (simple average)
        final_action = actions_np.mean(axis=0)

        next_state, _, _, _, _ = env.step(final_action)


        reward = maddpg.compute_reward(env)

        maddpg.replay_buffer.push(
            state,
            actions.detach().numpy(),
            reward,
            next_state
        )

        maddpg.update(batch_size=32)

        state = next_state

    print(f"Episode {ep+1} finished")
