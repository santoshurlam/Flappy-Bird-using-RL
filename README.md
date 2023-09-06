# Frozen Lake Environment

#### Created in [Frozen-Lake](https://github.com/RaviAgrawal-1824/Assignment-1-Frozen-Lake) environment.

## Requirements
To run this environment, you need to have the following libraries installed:
- numpy
- matplotlib
- gym

## Description
**Description**: For better understanding of the **Policy** and **Value** Iteration using the Frozen lake environment for both Deterministic and Stochastic of fully observable environments.

### Non-Slippery Environment

![](https://i.imgur.com/RlJjiZM.gif) ![](https://i.imgur.com/1dpekVN.gif)

### Slippery Environment

![](https://i.imgur.com/9dF44vt.gif)

This Frozen Lake environment is solved by Dynamic Programming Method using Reinforcement learning.

## Environment Description
  ### State Space
  - There are 16 states in 4x4 Environment and 64 states in 8x8 Environment
  - Each state has 0 as reward except terminal state
  - Any state may contain lake and the aim of the agent is to reach the Goal in optimal way using policy and value iteration
  - Dictionary contain every state with taking all 4 action and transition probability to take action with reaward getting.
  ### Action Space
  There are 4 actions for every state that agent can take,
  - Left - 0
  - Down - 1
  - Right - 2
  - Up - 3

## Algorithm
Here Dynamic Programming method is used for convergence of policy.
This can also be done by two ways
### Policy Iteration
  - Evaluating Value function for all states
  - Acting greedy toward policy using action value function evaluated using value function
  - Iterated many times upto convergence of policy
### Value Iteration
  - Evaluating Value function for particular state
  - Taking Greedy of all action it can take from that state using Action value function
  - Then converging policy


# Empty Room Environment


**Description**: To train agent to reach goal state by using different Algorithms,Directions of agent is also considerd.In 6x6 and 8x8 minigrid.


**Installation:**

pip install minigrid

**Action Space**
The action space Used here -

	Turn LEFT - 0
	Turn Right - 1
	Move Forward - 2
