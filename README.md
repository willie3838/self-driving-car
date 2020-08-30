# Autonomous Vehicle Framework Using Game Theory and Reinforcement Learning

This simulation was not created by me. Full credits should go towards @elurent for creating this open-source simulation: https://github.com/eleurent/highway-env

## Contributions
This simulation was modified to fit my research's goal of achieving an adaptive control strategy in a four-way unsignalized intersection. Aside from modifying the simulation, I implemented a deep reinforcement learning alogrithm (Deep-Q-Network) to train an agent to navigate through the intersection using two different policies: a level 1 and level 2 driver. A level 1 driver represents a passive driver that moves forward slowly prior to crossing an intersection while a level 2 driver represents an aggressive driver that proceeds through the intersection without hesitating. The Deep-Q-Network algorithm can be located [here](DQN/DQN_trainer.py) and the models that I trained can be located [here](models/intersection/creep). 


To view a full overview of what was modified and added to the simulation, please view my [technical paper](paper.pdf) 
