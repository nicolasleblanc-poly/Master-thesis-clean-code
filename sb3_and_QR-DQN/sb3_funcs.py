import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import gc
import panda_gym
from save_data_func import save_data

'''
The hyperparameters are from the sb3 Hugging face page:
https://huggingface.co/sb3/models
'''

class EpisodicReturnTrackerCallback(BaseCallback):
    def __init__(self, n_episodes, verbose=1):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        self.episodic_returns = []
        self.episodes = 0

    def _on_step(self) -> bool:
        # Check infos for episode end info (Monitor wrapper required!)
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episodes += 1
                ep_return = info["episode"]["r"]
                self.episodic_returns.append(ep_return)
                if self.verbose > 0:
                    print(f"Episode {self.episodes} return: {ep_return:.2f}")
        
        return self.episodes < self.n_episodes

    def _on_training_end(self):
        if self.verbose > 0:
            print(f"\nFinished training after {self.episodes} episodes.")

class ObservationOnlyWrapper(gym.Wrapper):
    """
    Wrapper that modifies Panda Gym environments to return only the observation component.
    This is useful for SB3 algorithms that don't need the full HER-style state dict.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Override observation_space to match the extracted observation
        self.observation_space = env.observation_space["observation"]
        
    def reset(self, **kwargs):
        """
        Reset the environment and return only the observation
        """
        obs_dict, info = self.env.reset(**kwargs)
        # print("obs_dict['observation']: ", obs_dict['observation'], "\n")
        return obs_dict['observation'], info #['observation']
    
    def step(self, action):
        """
        Take a step in the environment and return (obs, reward, done, info)
        with obs being just the observation component
        """
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        obs = obs_dict['observation']
        # print("obs_dict: ", obs_dict, "\n")
        return obs, reward, terminated, truncated, info


def run(env_seeds, prob, method_name, steps_per_episode, max_episodes):
    episodic_return_seeds = []
    
    """
    Fixed timesteps per episode for each environment.
    Only ran a single environment at a time
    No normalize, noise_std, noise_type, nor normalize_kwargs parameters
    No learning_starts parameter
    No action_noise parameter
    No env_wrapper parameter
    """
    
    for seed in env_seeds:
        if prob == "CartPole":
            # Hyperparameters were ran for v1 but v0 is the same
            # and my tests are with 200 steps per episode, so
            # I chose to use v0 since it is made for 200 steps per episode
            env = gym.make("CartPole-v0", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu', ent_coef=0.0) # , no normalize parameter
            # elif method_name == "DDPG":
            #     model = DDPG("MlpPolicy", env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu', batch_size=256, learning_rate='lin_0.001', ent_coef=0.0, gae_lambda=0.8, gamma=0.98, n_epochs=20, n_steps=32) # , clip_range='lin_0.2',  # , no normalize parameter
            # elif method_name == "SAC":
            #     model = SAC("MlpPolicy", env)
            # elif method_name == "TD3":
            #     model = TD3("MlpPolicy", env)
            # elif method_name == "TQC":
            #     model = TQC("MlpPolicy", env)
            
        elif prob == "Acrobot":
            # Hyperparameters were ran for v1
            env = gym.make("Acrobot-v1", render_mode="rgb_array")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu', ent_coef=0.0) # policy_kwargs={'norm_obs': True, 'norm_reward': False}, no normalize parameter
            # elif method_name == "DDPG":
            #     model = DDPG("MlpPolicy", env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu', ent_coef=0.0, gae_lambda=0.94, gamma=0.99, n_epochs=4, n_steps=256) # policy_kwargs={'norm_obs': True, 'norm_reward': False}, no normalize parameter
            # elif method_name == "SAC":
            #     model = SAC("MlpPolicy", env)
            # elif method_name == "TD3":
            #     model = TD3("MlpPolicy", env)
            # elif method_name == "TQC":
            #     model = TQC("MlpPolicy", env)
                    
        elif prob == "MountainCar":
            # Hyperparameters were ran for v0
            env = gym.make("MountainCar-v0", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C("MlpPolicy", env, device='cpu', ent_coef=0.0) # policy_kwargs={'norm_obs': True, 'norm_reward': False}, no normalize parameter
            # elif method_name == "DDPG":
            #     model = DDPG("MlpPolicy", env)
            elif method_name == "PPO":
                model = PPO("MlpPolicy", env, device='cpu', ent_coef=0.0, gae_lambda=0.98, gamma=0.99, n_epochs=4, n_steps=16) # policy_kwargs={'norm_obs': True, 'norm_reward': False}, , no normalize parameter
            # elif method_name == "SAC":
            #     model = SAC("MlpPolicy", env)
            # elif method_name == "TD3":
            #     model = TD3("MlpPolicy", env)
            # elif method_name == "TQC":
            #     model = TQC("MlpPolicy", env)
        
        elif prob == "LunarLander":
            # Hyperparameters were ran for v2
            env = gym.make("LunarLander-v2", render_mode="rgb_array")
            
            # env = gym.make("LunarLander-v3", render_mode="rgb_array")
            
            if method_name == "A2C": # Same hyperparameters as for v2 and for v3
                model = A2C(policy="MlpPolicy", env=env, device='cpu', ent_coef=1e-05, gamma=0.995, learning_rate='lin_0.00083', n_steps=5) # , no normalize parameter
            # elif method_name == "DDPG":
            #     model = DDPG("MlpPolicy", env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu', batch_size=64, ent_coef=0.01, gae_lambda=0.98, gamma=0.999, n_epochs=4, n_steps=1024) # , no normalize parameter
            # elif method_name == "SAC":
            #     model = SAC("MlpPolicy", env)
            # elif method_name == "TD3":
            #     model = TD3("MlpPolicy", env)
            # elif method_name == "TQC":
            #     model = TQC("MlpPolicy", env)
        
        elif prob == "Pendulum":
            env = gym.make("Pendulum-v1", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu', ent_coef=0.0, gae_lambda=0.9, gamma=0.99, learning_rate=7e-4, max_grad_norm=0.5, n_steps=8, normalize_advantage=False, policy_kwargs=dict(log_std_init=-2, ortho_init=False), use_rms_prop=True, use_sde=True, vf_coef=0.4) # policy_kwargs={'norm_obs': True, 'norm_reward': False}, no normalize parameter
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env, buffer_size=200000, gamma=0.98, gradient_steps=-1, learning_rate=0.001, policy_kwargs=dict(net_arch=[400,300]), train_freq=(1,'episode'), ) # learning_starts=10000, action_noise instead of noise_std=0.1, action_noise=0.1,
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu', clip_range=0.2, ent_coef=0.0, gae_lambda=0.95, gamma=0.9, learning_rate=0.001, n_epochs=10, n_steps=1024, sde_sample_freq=4, use_sde=True) # no normalize parameter
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env, learning_rate=0.001) # no normalize parameter
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env, buffer_size=200000, gamma=0.98, gradient_steps=-1, learning_rate=0.001, policy_kwargs=dict(net_arch=[400,300]), train_freq=(1,'episode')) # no learning_starts, no noise_std nor noise_type parameters, no normalize parameter
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env, learning_rate=0.001) # no normalize parameter
        
        elif prob == "InvertedPendulum": # No hyperparemeter tuning available from sb3 for this env
            env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env)
        
        elif prob == "MountainCarContinuous":
            env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu', ent_coef=0.0, n_steps=100, policy_kwargs=dict(log_std_init=0.0, ortho_init=False), sde_sample_freq=16, use_sde=True) # no normalize parameter, no normalize_kwargs
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env) # no noise_std, no noise_type, no normalize parameter
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu', batch_size=256, clip_range=0.1, ent_coef=0.00429, gae_lambda=0.9, gamma=0.9999, learning_rate=7.77e-05, max_grad_norm=5, n_epochs=10, n_steps=8, policy_kwargs=dict(log_std_init=-3.29, ortho_init=False), use_sde=True, vf_coef=0.19) # no normalize parameter, no normalize_kwargs
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env, batch_size=512, buffer_size=50000, ent_coef=0.1, gamma=0.9999, gradient_steps=32, learning_rate=0.0003, policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]), tau=0.01, train_freq=32, use_sde=True) # no learning_starts, no normalize parameter
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env) # no noise_std, no noise_type, no normalize parameter
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env, batch_size=512, buffer_size=50000, ent_coef=0.1, gamma=0.9999, gradient_steps=32, learning_rate=0.0003, policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]), tau=0.01, train_freq=32, use_sde=True) # no learning_starts, no normalize parameter
        
        elif prob == "LunarLanderContinuous":
            # Hyperparameters were ran for v2
            env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
            
            # env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu', ent_coef=0.0, gae_lambda=0.9, gamma=0.99, learning_rate=7e-4, max_grad_norm=0.5, n_steps=8, normalize_advantage=False, policy_kwargs=dict(log_std_init=-2, ortho_init=False), use_rms_prop=True, use_sde=True, vf_coef=0.4) # no normalize, no normalize_kwargs
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env, buffer_size=200000, gamma=0.98, gradient_steps=-1, learning_rate=0.001, policy_kwargs=dict(net_arch=[400,300]), train_freq=(1,'episode')) # no learning_starts, no noise_std nor noise_type parameters, no normalize parameter
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu', batch_size=64, ent_coef=0.01, gae_lambda=0.98, gamma=0.999, n_epochs=4, n_steps=1024) # no normalize parameter
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env, batch_size=256, buffer_size=1000000, ent_coef='auto', gradient_steps=1, learning_rate=7.3e-4, policy_kwargs=dict(net_arch=[400, 300]), tau=0.01, train_freq=1)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env, buffer_size=200000, gamma=0.98, gradient_steps=-1, learning_rate=0.001, policy_kwargs=dict(net_arch=[400,300]), train_freq=(1,'episode')) # no learning_starts, no noise_std nor noise_type parameters, no normalize parameter
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env, batch_size=256, buffer_size=1000000, ent_coef='auto', gradient_steps=1, learning_rate=7.3e-4, policy_kwargs=dict(net_arch=[400, 300]), tau=0.01, train_freq=1)
        
        elif prob == "Reacher":
            # No hyperparameter tuning available from sb3 for this env
            
            env = gym.make("Reacher-v5", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env)
        
        elif prob == "PandaReacher":
            # Hyperparameter tuning from sb3 is only available for TQC
            # I assume its for the PandaReach-v3 environment (dense)
            
            env = gym.make("PandaReach-v3", render_mode="rgb_array")
            env = ObservationOnlyWrapper(env)  # Wrap the environment to only return the observation
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env, batch_size=256, buffer_size=1000000, ent_coef='auto', gamma=0.95, learning_rate=0.001, policy_kwargs=dict(net_arch=[64,64], n_critics=1), replay_buffer_class='HerReplayBuffer', replay_buffer_kwargs=dict(online_sampling=True, goal_selection_strategy='future', n_sampled_goal=4)) # no normalize parameter
        
        elif prob == "PandaReacherDense":
            # No hyperparameter tuning available from sb3 for this env
            # For the dense version, only TQC has hyperparameter tuning available
            # I reuse the same hyperparameters for TQC here
            
            env = gym.make("PandaReachDense-v3", render_mode="rgb_array")
            env = ObservationOnlyWrapper(env)  # Wrap the environment to only return the observation
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env)
        
        elif prob == "CartPoleContinuous":
            # Use the same hyperparameters as for the discrete Cart Pole env
            
            import cartpole_continuous as cartpole_env
            env = cartpole_env.CartPoleContinuousEnv(render_mode="rgb_array")#.unwrapped
            # env = ObservationOnlyWrapper(env)  # Wrap the environment to only return the observation
            env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu', ent_coef=0.0, normalize_advantage=False)
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu', batch_size=256, clip_range=0.2, ent_coef=0.0, gae_lambda=0.8, gamma=0.98, learning_rate=0.001, n_epochs=20, n_steps=32, normalize_advantage=False)
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env)
        
        elif prob == "PandaPusher":
            # No hyperparameter tuning available from sb3 for this env
            # For the dense version, only TQC has hyperparameter tuning available
            # I reuse the same hyperparameters for TQC here
            
            env = gym.make("PandaPush-v3", render_mode="rgb_array")
            env = ObservationOnlyWrapper(env)  # Wrap the environment to only return the observation
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env)
             
        elif prob == "PandaPusherDense":
            # No hyperparameter tuning available from sb3 for this env
            # For the dense version, only TQC has hyperparameter tuning available
            # I reuse the same hyperparameters for TQC here
            
            env = gym.make("PandaPushDense-v3", render_mode="rgb_array")
            env = ObservationOnlyWrapper(env)  # Wrap the environment to only return the observation
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env)
                
        elif prob == "MuJoCoPusher":
            # No hyperparameter tuning available from sb3 for this env
            
            env = gym.make("Pusher-v5", render_mode="rgb_array")
            
            if method_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "DDPG":
                model = DDPG(policy="MlpPolicy", env=env)
            elif method_name == "PPO":
                model = PPO(policy="MlpPolicy", env=env, device='cpu')
            elif method_name == "SAC":
                model = SAC(policy="MlpPolicy", env=env)
            elif method_name == "TD3":
                model = TD3(policy="MlpPolicy", env=env)
            elif method_name == "TQC":
                model = TQC(policy="MlpPolicy", env=env)

        env = DummyVecEnv([lambda: env])  # Required for SB3
        env = VecMonitor(env, "logs/")  # Saves logs to "logs/"
        env.seed(seed)  # Set the seed for reproducibility
        print("method ", method_name, "\n")

        # Initialize callback
        return_logger = EpisodicReturnTrackerCallback(n_episodes=max_episodes)
        
        model.learn(total_timesteps=steps_per_episode*max_episodes, callback=return_logger)
         
        episodic_return_seeds.append(return_logger.episodic_returns)
        print("len(return_logger.episodic_returns): ", len(return_logger.episodic_returns))
        
        env.close()
        del env
        del model
        gc.collect()
        
    episodic_return_seeds = np.array(episodic_return_seeds)

    mean_episodic_return = np.mean(episodic_return_seeds, axis=0)
    std_episodic_return = np.std(episodic_return_seeds, axis=0)
    
    save_data(prob, method_name, episodic_return_seeds, mean_episodic_return, std_episodic_return)
    
    
    
    
