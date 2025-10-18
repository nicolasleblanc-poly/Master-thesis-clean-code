import numpy as np

def generate_random_action_sequences(prob_vars):
    # Generate new action sequences by sampling uniformly

    if prob_vars.prob_name == "Pendulum" or prob_vars.prob_name == "Cartpole":
        particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.horizon))

    return particles
