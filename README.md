# Master thesis clean code

This repository contains our 21 MPC methods, as well as MFRL (A2C, PPO, SAC, DDPG, TD3, TQC), and trajectory optimization methods (CEM, iCEM, MPPI) from the literature.

Our 21 MPC methods consist in using:
- Generating initial action sequences
## e23e

Let's go over the different elements of this repository:
- I used python 3.11 for the code in this repository. USe pip install -r pathto/requirements_py311.txt (replace the pathto with your path to the file)
- The GP-MPC folder contains the GP-MPC code. GP-MPC uses python 3.10. See this repository of mine for the code and the requirements.txt file to run it:
- The PETS folder contains the PETS-CEM code. PETS-CEM uses python 3.10. See this repository of mine for the code and the requirements.txt file to run it:
- There is a ..._analysis.ipynb file for each of the environments I have run (Acrobot, discrete Cart Pole, discrete Mountain Car, discrete Lunar Lander, continuous Cart Pole, Pendulum, Inverted Pendulum, continuous Mountain Car, continuous Lunar Lander, MuJoCo Reacher, Panda Gym Reach (dense and sparse rewards). I have written code and have some results for the MuJoCo Pusher and Panda Gym Push (dense and sparse rewards). The pusher environments didn't work well, so further work needs to be done to understand why. I tried running Panda Gym Push using the model of the environment, but these tests crashed because of an out-of-memory error.
