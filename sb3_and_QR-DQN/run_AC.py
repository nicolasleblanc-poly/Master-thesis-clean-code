from sb3_funcs import run
   
env_seeds = [0, 8, 15]

steps_per_episode = 200
max_episodes = 600
prob = "Acrobot"
# A2C
run(env_seeds, prob, "A2C", steps_per_episode, max_episodes)
# PPO
run(env_seeds, prob, "PPO", steps_per_episode, max_episodes)
# # DDPG
# run("CartPole", "DDPG", steps_per_episode, max_episodes)
# # SAC
# run("CartPole", "SAC", steps_per_episode, max_episodes)
# # TD3
# run("CartPole", "TD3", steps_per_episode, max_episodes)
# # TQC
# run("CartPole", "TQC", steps_per_episode, max_episodes)


