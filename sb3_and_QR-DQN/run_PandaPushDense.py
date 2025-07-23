from sb3_funcs import run

env_seeds = [0, 8, 15]

steps_per_episode = 50
max_episodes = 400
prob = "PandaPusherDense"
# A2C
run(env_seeds, prob, "A2C", steps_per_episode, max_episodes)
# PPO
run(env_seeds, prob, "PPO", steps_per_episode, max_episodes)
# DDPG
run(env_seeds, prob, "DDPG", steps_per_episode, max_episodes)
# SAC
run(env_seeds, prob, "SAC", steps_per_episode, max_episodes)
# TD3
run(env_seeds, prob, "TD3", steps_per_episode, max_episodes)
# TQC
run(env_seeds, prob, "TQC", steps_per_episode, max_episodes)


