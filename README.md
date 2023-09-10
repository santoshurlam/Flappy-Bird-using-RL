# Frozen Lake Environment

#### Created in [Frozen-Lake](https://github.com/RaviAgrawal-1824/Assignment-1-Frozen-Lake) environment.

## Requirements
To run this environment, you need to have the following libraries installed:
- numpy
- matplotlib
- gymnasium

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

#### Created in [MiniGrid-Empty-Environment](https://github.com/Farama-Foundation/MiniGrid) environment.

![](https://i.imgur.com/3m9a615.gif) ![](https://i.imgur.com/ahGLjM7.gif)

**Description**: To train agent to reach goal state by using different Algorithms,Directions of agent is also considerd.In 6x6 and 8x8 minigrid.

## Requirements
To run this environment, you need to have the following libraries installed:
- numpy
- matplotlib
- gymnasium
- minigrid

## Installation
Use this code for intalling some library
- pip install minigrid
- pip install numpy
- pip install matplotlib
- pip install gymnasium

**Action Space**
The action space Used here -

	Turn LEFT - 0
	Turn Right - 1
	Move Forward - 2
**State Space**
* There are 16 states in MiniGrid-Empty-6x6-v0 environment and each cell is represented by (x,y) where x = 1,2,3,4 and y = 1,2,3,4 and also the agent position can be accessed through the built-in function called "agent_pos".
* Similarily, there are 36 states in MiniGrid-Empty-8x8-v0 environmnet where each cell is represnted by (x,y) where x = 1 to 6 & y = 1 to 6.
* State space also requires the direction of the agent facing towards that can be accessed through user built-in functions from the minigrid files which is "agent_dir".
* Agent aim is to reach the final goal state in an optimized way by using the algorithms like Monte-carlo, SARSA, SARSA Lambda, Q-Learning.

### Rewards
Goal state has reward 1 otherwise 0 everywhere.

## Algorithms used in this Environment
Four algorithm are used to converge the policy and take optimal actions,
- Monte-Carlo
- SARSA
- SARSA Lambda
- Q-Learning

## Results

### MiniGrid-Empty-6x6-v0

![Imgur](https://i.imgur.com/XAtvwPw.png)

![Imgur](https://i.imgur.com/cH8jTrB.png)

### MiniGrid-Empty-8x8-v0

![Imgur](https://i.imgur.com/ccNcDDY.png)

![Imgur](https://i.imgur.com/oyboRlw.png)

