# Master thesis clean code

This repository contains our 21 MPC methods, as well as MFRL (A2C, PPO, SAC, DDPG, TD3, TQC), and trajectory optimization methods (CEM, iCEM, MPPI) from the literature.

## Our 21 MPC methods consist in using:
### Generating initial action sequences: 
- Sampling a uniform distribution

- Sampling an action sequence neural network (ASNN)

### Model of the environment: 
- Quantile regression neural network (QRNN)
-- dw
- 50NN
- MSENN

### Ways to optimize action sequences in MPC:
- Particle filtering (PF)
- Cross-entropy method (CEM)

### Methods to change/generate action sequences before the next step in the environment: 
- Replace the previous iteration's action sequences with new ones sampled from a uniform distribution
- Shift the action sequences to remove the action taken and replace the vacant action by one sampled from a uniform distribution
- Shift the action sequences to remove the action taken and replace the vacant action by one sampled from the ASNN

## Let's go over the different elements of this repository:
- I used python 3.11 for the code in this repository. USe pip install -r pathto/requirements_py311.txt (replace the pathto with your path to the file)
- The AUC_data folder contains the different means and standard deviations of the area under the curve (AUC) of the episodic returns for the different algorithms and tasks. The AllAlgoComparison.ipynb file allows for normalizing of the AUC data and ranking the algorithms over the different discrete and continuous action space environments.
- The MPC_methods folder contains the code for our 21 MPC methods.
- The GP-MPC folder contains the GP-MPC code. GP-MPC uses python 3.10. See this repository of mine for the code and the requirements.txt file to run it:
- The PETS folder contains the PETS-CEM code. PETS-CEM uses python 3.10. See this repository of mine for the code and the requirements.txt file to run it:
- There is a ..._analysis.ipynb file for each of the environments I have run (Acrobot, discrete Cart Pole, discrete Mountain Car, discrete Lunar Lander, continuous Cart Pole, Pendulum, Inverted Pendulum, continuous Mountain Car, continuous Lunar Lander, MuJoCo Reacher, Panda Gym Reach (dense and sparse rewards). I have written code and have some results for the MuJoCo Pusher and Panda Gym Push (dense and sparse rewards). The pusher environments didn't work well, so further work needs to be done to understand why. I tried running Panda Gym Push using the model of the environment, but these tests crashed because of an out-of-memory error.
