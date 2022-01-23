[//]: # (Image References)


# CartPole via PPO - Evaluated with sticky actions

## Description

main.ipynb trains an agent in CartPole-v0 (OpenAI Gym) environment via Proximal Policy Optimization (PPO) algorithm with GAE. Evaluation is carried out in both standard way and with sticky actions.

A reward of **+1** is provided for every step taken, and a reward of **0** is provided at the termination step. The state space has **4** dimensions and contains the cart position, velocity, pole angle and pole velocity at tip. 
Given this information, the agent has to learn how to select best actions. 
Two discrete actions are available, corresponding to:

- **`0`** - 'Push cart to the left'
- **`1`** - 'Push cart to the right'

## Existing Problems

- used an internel clock to measure the time spend on computations and pass this information onto the environment. The issue with this approach is that it creates a synchronous paradigm where state and actions can not be easily shared asynchronously.
- seems to be some problems with wandb.log({"total_rewards": mean_rewards})
- repeat = level * int((t2 - t1)/(t3 - t2)) seems to be pretty large and PPO behaves poorly with sticky actions