import os
import numpy as np

def save_data(prob, method_name, episodic_rep_returns, mean_episodic_returns, std_episodic_returns):
    
    # Get the folder where this script is located
    origin_folder = os.path.dirname(os.path.abspath(__file__))
    # Construct full path to save
    save_path = os.path.join(origin_folder, f"{prob}_{method_name}_results.npz")

    np.savez(
        save_path,
        episode_rewards=episodic_rep_returns,
        mean_rewards=mean_episodic_returns,
        std_rewards=std_episodic_returns
        )