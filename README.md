# Master thesis clean code

This repository contains our 21 MPC methods, as well as MFRL (A2C, PPO, SAC, DDPG, TD3, TQC, QR-DQN), and trajectory optimization methods (CEM, iCEM, MPPI) from the literature. Other algorithms not in this repository are GP-MPC, PETS-CEM, DQN, and IV-DQN.

See the following repositories for the code and requirements.txt files to run:
- The GP-MPC folder contains the GP-MPC code. GP-MPC uses python 3.10. See this repository of mine for the code and the requirements.txt file to run it: https://github.com/nicolasleblanc-poly/GP-MPC-from-paper-for-my-Master-thesis/tree/main
- The PETS folder contains the PETS-CEM code. PETS-CEM uses python 3.10. See this repository of mine for the code and the requirements.txt file to run it: https://github.com/nicolasleblanc-poly/PETS-CEM-from-papers-for-my-Master-thesis/tree/main
- DQN, IV-DQN, and QR-DQN: https://github.com/nicolasleblanc-poly/DQN-and-IV-DQN-from-paper-for-Master-thesis

See my report here for more information on the résumé presented below: link

## We test 4 different components of an MPC method to generate 21 algorithms: 

### Generating initial action sequences: 
- Sampling a uniform distribution

- Sampling an action sequence neural network (ASNN)

### Model of the environment:
- Quantile regression neural network (QRNN)
- 50NN
- MSENN

### Ways to optimize action sequences in MPC:
- Particle filtering (PF)
- Cross-entropy method (CEM)

### Methods to change/generate action sequences before the next step in the environment: 
- Replace the previous iteration's action sequences with new ones sampled from a uniform distribution
- Shift the action sequences to remove the action taken and replace the vacant action with one sampled from a uniform distribution
- Shift the action sequences to remove the action taken and replace the vacant action with one sampled from the ASNN

## Our 21 MPC methods
For all the algorithms below, you can pick a model among QRNN, 50NN, and MSENN, as well as PF or CEM where they are mentionned.

### QRNN/50NN/MSENN-ASNN-PF/CEM
- Use the ASNN to generate action sequences at the start of an episode
- Shift the action sequences and replace the vacant action by one sampled from the ASNN

### QRNN/50NN/MSENN-basic-PF/CEM
- Same idea as the ASNN algorithms, but a uniform distribution is sampled instead of the ASNN.

### QRNN/50NN/MSENN-RS
- Generate new action sequences at each step in the environment by sampling a uniform distribution.

### QRNN/50NN/MSENN-rnd-PF/CEM
- Same idea as RS, but we optimize the action sequences in MPC for the same amount of iterations as the ASNN and basic methods.

## Let's go over the different elements of this repository:
- I used python 3.11 for the code in this repository. USe pip install -r pathto/requirements_py311.txt (replace the pathto with your path to the file)
- The AUC_data folder contains the different means and standard deviations of the area under the curve (AUC) of the episodic returns for the different algorithms and tasks. The AllAlgoComparison.ipynb file allows for normalizing of the AUC data and ranking the algorithms over the different discrete and continuous action space environments.
- The MPC_methods folder contains the code for our 21 MPC methods.
- There is a ..._analysis.ipynb file for each of the environments I have run (Acrobot, discrete Cart Pole, discrete Mountain Car, discrete Lunar Lander, continuous Cart Pole, Pendulum, Inverted Pendulum, continuous Mountain Car, continuous Lunar Lander, MuJoCo Reacher, Panda Gym Reach (dense and sparse rewards). I have written code and have some results for the MuJoCo Pusher and Panda Gym Push (dense and sparse rewards). The pusher environments didn't work well, so further work needs to be done to understand why. I tried running Panda Gym Push using the model of the environment, but these tests crashed because of an out-of-memory error.
