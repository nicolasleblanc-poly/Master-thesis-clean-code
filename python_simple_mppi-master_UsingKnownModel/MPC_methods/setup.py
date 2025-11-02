import numpy as np
import torch
import gymnasium as gym
# import panda_gym

class setup_class:
    def __init__(self, prob, prob_name, delta_t, sim_steps, state_cost_weight, terminal_cost_weight, do_RS, use_sampling, use_mid, use_QRNN, use_50NN, use_MSENN, model_state, optimizer_state, loss_state, replay_buffer_state, use_ASNN, model_ASNN, replay_buffer_ASNN, use_CEM=False, num_quantiles=None):

        self.prob = prob

        # self.prob_name = prob_name
        # self.delta_t = delta_t
        # self.max_steps = sim_steps
        # self.stage_cost_weight = state_cost_weight
        # self.terminal_cost_weight = terminal_cost_weight

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

        if self.prob_name == "Pendulum_TrueMPC":
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

            self.stage_cost_weight = torch.tensor([1.0, 0.1]) # weight for [theta, theta_dot]
            self.terminal_cost_weight = 5.0 * torch.tensor([1.0, 0.1]) # weight for [theta, theta_dot]


            def compute_state_cost_Pendulum_TrueMPC_1D(self, x_t) -> float:
                """calculate stage cost"""
                theta, theta_dot = x_t[0], x_t[1]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]
                stage_cost = (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2)#.sum()
                return stage_cost
            
            def compute_state_cost_Pendulum_TrueMPC_1D(self, x_t) -> float:
                """calculate stage cost"""
                # parse x_t

                # print("x_t.shape ", x_t.shape, "\n")

                theta, theta_dot = x_t[0], x_t[1]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                stage_cost = (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2)#.sum()
                return stage_cost

            def compute_state_cost_Pendulum_TrueMPC(self, x_t) -> float:
                """calculate stage cost"""
                # parse x_t

                # print("x_t.shape ", x_t.shape, "\n")

                theta, theta_dot = x_t[:,0], x_t[:,1]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                # print("theta.shape ", theta.shape, "\n")
                # print("theta_dot.shape ", theta_dot.shape, "\n")

                # print("self.stage_cost_weight[0][0]*theta**2 ", self.stage_cost_weight[0]*theta**2, "\n")
                # print("self.stage_cost_weight[0][1]*theta_dot**2 ", self.stage_cost_weight[1]*theta_dot**2, "\n")

                # print("self.stage_cost_weight[0][0]*theta**2 + self.stage_cost_weight[0][1]*theta_dot**2 ", (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2).shape, "\n")

                # print("(self.stage_cost_weight[0][0]*theta**2 + self.stage_cost_weight[0][1]*theta_dot**2).sum() ", (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2).shape, "\n") # .sum()

                # calculate stage cost
                # stage_cost = self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2
                stage_cost = (self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2)#.sum()
                return stage_cost

            def compute_terminal_cost_Pendulum_TrueMPC(self, x_T) -> float:
                """calculate terminal cost"""
                # parse x_T
                theta, theta_dot = x_T[:,0], x_T[:,1]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                # print("terminal_cost = self.terminal_cost_weight[0][0]*theta**2 + self.terminal_cost_weight[0][1]*theta_dot**2 ", (self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2).shape, "\n")

                # calculate terminal cost
                # terminal_cost = self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2
                terminal_cost = (self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2)#.sum()
                return terminal_cost

            self.compute_state_cost_Pendulum_1D = compute_state_cost_Pendulum_TrueMPC_1D
            self.compute_state_cost_Pendulum = compute_state_cost_Pendulum_TrueMPC
            self.compute_terminal_cost_Pendulum = compute_terminal_cost_Pendulum_TrueMPC

        if self.prob_name == "Cartpole_TrueMPC":
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
            self.action_low = -self.prob.max_force_abs
            self.action_high = self.prob.max_force_abs

            self.goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-2.4, -np.pi, -3, -10])
            self.states_high = torch.tensor([2.4, np.pi, 3, 10])

            def compute_state_cost_Cartpole_TrueMPC_1D(self, x_t) -> float:
            # def _c(self, x_t: np.ndarray) -> float:
                """calculate stage cost"""
                # parse x_t
                x, x_dot = x_t[0], x_t[2]
                theta, theta_dot = x_t[1], x_t[3]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                # calculate stage cost # (np.cos(theta)+1.0)
                stage_cost = self.stage_cost_weight[0]*x**2 + self.stage_cost_weight[1]*theta**2 + self.stage_cost_weight[2]*x_dot**2 + self.stage_cost_weight[3]*theta_dot**2
                return stage_cost

            def compute_state_cost_Cartpole_TrueMPC(self, x_t) -> float:
            # def _c(self, x_t: np.ndarray) -> float:
                """calculate stage cost"""
                # parse x_t
                x, x_dot = x_t[:, 0], x_t[:, 2]
                theta, theta_dot = x_t[:, 1], x_t[:, 3]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                # calculate stage cost # (np.cos(theta)+1.0)
                stage_cost = self.stage_cost_weight[0]*x**2 + self.stage_cost_weight[1]*theta**2 + self.stage_cost_weight[2]*x_dot**2 + self.stage_cost_weight[3]*theta_dot**2
                return stage_cost

            def compute_terminal_cost_Cartpole_TrueMPC(self, x_T) -> float:
                """calculate terminal cost"""
                # parse x_T
                x, x_dot = x_T[:, 0], x_T[:, 2]
                theta, theta_dot = x_T[:, 1], x_T[:, 3]
                theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

                # calculate terminal cost # (np.cos(theta)+1.0)
                terminal_cost = self.terminal_cost_weight[0]*x**2 + self.terminal_cost_weight[1]*theta**2 + self.terminal_cost_weight[2]*x_dot**2 + self.terminal_cost_weight[3]*theta_dot**2
                return terminal_cost

            self.compute_state_cost_Cartpole_1D = compute_state_cost_Cartpole_TrueMPC_1D
            self.compute_state_cost_Cartpole = compute_state_cost_Cartpole_TrueMPC
            self.compute_terminal_cost_Cartpole = compute_terminal_cost_Cartpole_TrueMPC
            
        if self.prob_name == "Pathtracking":
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
            self.action_low = np.array([-self.prob.max_steer_abs, -self.prob.max_accel_abs])
            self.action_high = np.array([self.prob.max_steer_abs, self.prob.max_accel_abs])

            self.goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-100, -5, -np.pi, -3, 0])
            self.states_high = torch.tensor([100, 5, np.pi, 3, 20])
            
            def compute_state_cost_Pathtracking_1D(self, x_t) -> float:
            # def _c(self, x_t: np.ndarray) -> float:
                """calculate stage cost"""
                # parse x_t
                # x, y, yaw, v = x_t
                x = x_t[0]
                y = x_t[1]
                yaw = x_t[2]
                v = x_t[3]
                yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

                # calculate stage cost
                _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
                stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                            self.stage_cost_weight[2]*(yaw-ref_yaw)**2 + self.stage_cost_weight[3]*(v-ref_v)**2
                return stage_cost

            def compute_state_cost_Pathtracking(self, x_t) -> float:
            # def _c(self, x_t: np.ndarray) -> float:
                """calculate stage cost"""
                # parse x_t
                # x, y, yaw, v = x_t
                x = x_t[:, 0]
                y = x_t[:, 1]
                yaw = x_t[:, 2]
                v = x_t[:, 3]
                yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

                # calculate stage cost
                _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
                stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                            self.stage_cost_weight[2]*(yaw-ref_yaw)**2 + self.stage_cost_weight[3]*(v-ref_v)**2
                return stage_cost

            def compute_terminal_cost_Pathtracking(self, x_T) -> float:
                """calculate terminal cost"""
                # # parse x_T
                # x, y, yaw, v = x_T
                x = x_T[:, 0]
                y = x_T[:, 1]
                yaw = x_T[:, 2]
                v = x_T[:, 3]
                yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

                # calculate terminal cost
                _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
                terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                                self.terminal_cost_weight[2]*(yaw-ref_yaw)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2
                return terminal_cost

            self.compute_state_cost_Pathtracking_1D = compute_state_cost_Pathtracking_1D
            self.compute_state_cost_Pathtracking = compute_state_cost_Pathtracking
            self.compute_terminal_cost_Pathtracking = compute_terminal_cost_Pathtracking

    def compute_state_cost_1D(self, prob, states):
        if prob == "Pendulum":
            return self.compute_state_cost_Pendulum_1D(self,states)
        elif prob == "Cartpole":
            return self.compute_state_cost_Cartpole_1D(self,states)
        elif prob == "PathTracking":
            return self.compute_state_cost_Pathtracking_1D(self,states)
        else:
            raise ValueError("Unknown problem")    

    def compute_state_cost(self, prob, states):
        if prob == "Pendulum":
            return self.compute_state_cost_Pendulum(self,states)
        elif prob == "Cartpole":
            return self.compute_state_cost_Cartpole(self,states)
        elif prob == "PathTracking":
            return self.compute_state_cost_Pathtracking(self,states)
        else:
            raise ValueError("Unknown problem")

    def compute_terminal_cost(self, prob, states):
        if prob == "Pendulum":
            return self.compute_terminal_cost_Pendulum(self,states)
        elif prob == "Cartpole":
            return self.compute_terminal_cost_Cartpole(self,states)
        elif prob == "PathTracking":
            return self.compute_terminal_cost_Pathtracking(self,states)
        else:
            raise ValueError("Unknown problem")

