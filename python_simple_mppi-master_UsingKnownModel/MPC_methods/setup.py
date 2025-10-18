import numpy as np
import torch
import gymnasium as gym
# import panda_gym

class setup_class:
    def __init__(self, prob, prob_name, delta_t, sim_steps, state_cost_weight, terminal_cost_weight, do_RS, use_sampling, use_mid, use_QRNN, use_50NN, use_MSENN, model_state, optimizer_state, loss_state, replay_buffer_state, use_ASNN, model_ASNN, replay_buffer_ASNN, use_CEM=False, num_quantiles=None):

        self.prob = prob

        self.prob_name = prob_name
        self.delta_t = delta_t
        self.max_steps = sim_steps
        self.stage_cost_weight = state_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        self.do_RS = do_RS
        self.use_sampling = use_sampling
        self.use_mid = use_mid
        self.use_QRNN = use_QRNN
        self.use_50NN = use_50NN
        self.use_MSENN = use_MSENN
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.loss_state = loss_state
        self.replay_buffer_state = replay_buffer_state
        self.use_ASNN = use_ASNN
        # self.loss_ASNN = loss_ASNN
        self.replay_buffer_ASNN = replay_buffer_ASNN
        self.model_ASNN = model_ASNN

        self.nb_actions = None

        # Generate random seeds
        self.random_seeds = [0, 8, 15]
        # print("random_seeds ", type(random_seeds[0]), "\n")
        self.nb_rep_episodes = len(self.random_seeds)

        self.laplace_alpha = 1
        self.use_CEM = use_CEM

        self.num_quantiles = num_quantiles

        self.nb_MPC_iters = 5

        # Constants
        self.batch_size = 32
        self.num_particles = 100
        self.quantiles = torch.tensor(np.array([0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), dtype=torch.float32)
        self.num_quantiles = len(self.quantiles)
        self.nb_reps_MPC = 10
        self.nb_random = 0 # 10

        if self.prob_name == "Pendulum":
            self.discrete = False
            self.horizon = 20
            # self.max_episodes = 300
            # self.max_steps = 200

            # For test
            # self.max_episodes = 2
            # self.max_steps = 3

            # Current test values
            # self.std = 0
            self.std = 3e-1
            # self.std = 1.5

            # Older test values
            # std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            self.change_prob = None

            # self.std_string = "0"
            self.std_string = "3em1"
            # std_string = "15"
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            # Hyperparameters
            self.state_dim = len(prob.state)
            # self.env.observation_space.shape[0]
            # state_dim = env.observation_space.shape[0]-1 # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = -self.prob.max_torque
            self.action_high = self.prob.max_torque
            
            self.goal_state = torch.tensor([0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-np.pi, -self.prob.max_speed])
            self.states_high = torch.tensor([np.pi, self.prob.max_speed])

            def compute_state_cost_Pendulum(self, x_t) -> float:
                """calculate stage cost"""
                # parse x_t

                print("x_t.shape ", x_t.shape, "\n")

                theta, theta_dot = x_t[:,0], x_t[:,1]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                print("theta.shape ", theta.shape, "\n")
                print("theta_dot.shape ", theta_dot.shape, "\n")

                print("self.stage_cost_weight[0][0]*theta**2 ", self.stage_cost_weight[0]*theta**2, "\n")
                print("self.stage_cost_weight[0][1]*theta_dot**2 ", self.stage_cost_weight[1]*theta_dot**2, "\n")

                print("self.stage_cost_weight[0][0]*theta**2 + self.stage_cost_weight[0][1]*theta_dot**2 ", (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2).shape, "\n")

                print("(self.stage_cost_weight[0][0]*theta**2 + self.stage_cost_weight[0][1]*theta_dot**2).sum() ", (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2).shape, "\n") # .sum()

                # calculate stage cost
                # stage_cost = self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2
                stage_cost = (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2)#.sum()
                return stage_cost

            def compute_terminal_cost_Pendulum(self, x_T) -> float:
                """calculate terminal cost"""
                # parse x_T
                theta, theta_dot = x_T[:,0], x_T[:,1]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                print("terminal_cost = self.terminal_cost_weight[0][0]*theta**2 + self.terminal_cost_weight[0][1]*theta_dot**2 ", (self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2).shape, "\n")

                # calculate terminal cost
                # terminal_cost = self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2
                terminal_cost = (self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2)#.sum()
                return terminal_cost
            
            self.compute_state_cost_Pendulum = compute_state_cost_Pendulum
            self.compute_terminal_cost_Pendulum = compute_terminal_cost_Pendulum

    def compute_state_cost(self, prob, states):
        if prob == "Pendulum":
            return self.compute_state_cost_Pendulum(self,states)
        else:
            raise ValueError("Unknown problem")

    def compute_terminal_cost(self, prob, states):
        if prob == "Pendulum":
            return self.compute_terminal_cost_Pendulum(self,states)
        else:
            raise ValueError("Unknown problem")

