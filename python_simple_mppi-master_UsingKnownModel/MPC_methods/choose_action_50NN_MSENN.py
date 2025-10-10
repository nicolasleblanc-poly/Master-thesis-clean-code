import numpy as np
import random
import torch

# from mpc import mpc_func
from mpc_50NN_MSENN import mpc_50NN_MSENN_func
from particle_filtering import particle_filtering_func, discrete_cem_func, continuous_cem_func

def choose_action_func_50NN_MSENN(prob_vars, state, particles, episode=0, step=1, goal_state=None):

    best_cost = float('inf')
    best_action_sequence = None

    nb_reps = prob_vars.nb_MPC_iters

    if prob_vars.do_RS:
        nb_reps = 1

    for rep in range(nb_reps):
    
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(prob_vars.num_particles, 1)

        costs = mpc_50NN_MSENN_func(prob_vars, sim_states, particles)

        min_idx = torch.argmin(costs)
        
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_action_sequence = particles[min_idx].copy()
            best_first_action = best_action_sequence[0].item()

        if best_cost == None or best_action_sequence is None: # For debugging
            best_cost = costs[0]
            best_action_sequence = particles[0].copy()
            print("costs ", costs, "\n")
            print("best_action_sequence ", best_action_sequence, "\n")
            print("sim_states ", sim_states, "\n")
            print("state ", state, "\n")
            print("particles ", particles, "\n")

        if not prob_vars.do_RS:
            if prob_vars.use_CEM:
                best_first_action, particles = continuous_cem_func(prob_vars, particles, costs, best_action_sequence)

            else:
                best_first_action, particles = particle_filtering_func(prob_vars, particles, costs, best_action_sequence)

    return best_first_action, best_action_sequence, best_cost, particles

