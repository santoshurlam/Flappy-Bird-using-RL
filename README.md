
# Frozen Lake Environment

**Aim:**
* To achieve the desired outcome efficiently, the agent needs to find the most effective route using dynamic programming.

**Description:**

* This environment is composed of grids in 4x4 and 8x8 sizes. Each grid can either be a frozen lake or a hole, and the objective is to reach the final grid containing a gift.
* It is a model-based Environment.


### Deterministic Environment

![](https://i.imgur.com/RlJjiZM.gif) ![](https://i.imgur.com/1dpekVN.gif)

### Stochastic Environment

![](https://i.imgur.com/9dF44vt.gif)


## State Space
* For a 4x4 grid, each cell or state is represented by an integer from 0 to 15. For an 8x8 grid, the range is from 0 to 63.
* If an agent takes an action towards the grid boundary, it remains in the same state.

## Action Space

In any given state, an agent can take various actions

```bash
  Left - 0
  Down - 1
  Right- 2
  Up   - 3
```

## Reward
* If the agent falls into the hole or lands on a frozen lake, the reward is 0.
* However, if it reaches the goal state, it receives a reward of 1.

## Algorithms
The Dynamic Programming method is utilized to achieve policy convergence.
There are two alternative methods to accomplish this task.
#### Policy Iteration
  - Computing value function for all states.
  - Using the action value function to evaluate policy with greediness.
  - Continue iterating until the policy reaches convergence.
#### Value Iteration
  - A technique for determining the best value function through iterative updates of the Bellman equation.
  - Taking the best action for a state using the action-value function.
  - Continue the iteration until the policy converges.

## Results

#### FrozenLake-v1 (Deterministic)

![Deterministic](https://i.imgur.com/zBB5PSV.png)

#### FrozenLake-v1 (Stochastic)

![Stochastic](https://i.imgur.com/bPGPDrX.png)

# Minigrid Environment

**Aim:**

* The objective for the agent is to achieve the goal state in the most efficient manner possible.

**Description:**

* The Minigrid Environment is an empty room containing one agent and one goal state, with no obstacles.
* There are two environments available: MiniGrid-Empty-6x6-v0 and MiniGrid-Empty-8x8-v0.
* The environment is model-free.


![](https://i.imgur.com/4lCwL8g.gif) ![](https://i.imgur.com/tIZ0FNG.gif)

## State Space

- Each state in MiniGrid-Empty-6x6-v0 is represented by (x,y) coordinates, where x and y range from 1 to 4 among the 16 states. 
- In the MiniGrid-Empty-8x8-v0 environment, there are 36 states represented by (x,y) coordinates where x and y range from 1 to 6.
- The state space includes the direction of the agent, which is indicated as follows:
  	- 0 - Right 
  	- 1 - Down
  	- 2 - Left
  	- 3 - Up
- The observation includes an image array that can be utilized to locate the agent within the environment.

## Action Space

An agent can take three actions to alter its state,

```bash
  - 0 - Turn Left
  - 1 - Turn Right
  - 2 - Move Forward

```

## Rewards

* Success earns a reward of '1 - 0.9 * (step_count / max_steps)' while failure earns '0'.
* 'max_steps' refers to the maximum number of steps an agent can take in an episode.
* The 'step_count' records the number of steps taken by the agent during an episode, but it cannot exceed the 'max_steps' limit.

## Algorithms
```bash
 Monte-Carlo
 SARSA
 SARSA Lambda
 Q-Learning
```
## Results
#### MiniGrid-Empty-6x6-v0
![Graph 1](https://i.imgur.com/spQjmOb.png)

![Graph 2](https://i.imgur.com/kXaOTd5.png)
# Flappy Bird Environment

**Aim:**
* The agent bird learns to score by crossing pipes with the Q-Learning Algorithm.

**Description:**

* The game has a bird as the agent and randomly generated pipes. The bird can only move vertically while the pipes move horizontally. There is also a base and background.
* The Flappy Bird Environment is a model-free environment.

<p align = "center">
    <img src = "https://i.imgur.com/ZgW3wYP.gif" alt = "Flappy bird">
</p>



## Requirements
* Matplotlib
* NumPy
* flappy_bird_gym (Cloned Repository from [Flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym))

```bash
  pip install Matplotlib
  pip install NumPy
```
**Note:** An algorithm Python file was created in the cloned repository folder to import flappy_bird_gym directly into the code.

## State Space

* The environment comprises the bird (agent) center's location as its state.
* Location shows bird's distance from the next pipe and lower pipe's hole center.
* The state keeps resetting every time I hit the pipe or crash on the base.
* Agent moves upward direction only if it flaps since the PLAYER_VEL_ROT = 0 and player_rot = 0 degrees but initially it was 45 degrees.

## Action Space

Here are the possible moves that the agent can make at any given state:

```bash
  Flap       - 1
  Do nothing - 0
```
## Reward 

* When the bird crosses the pipes, it earns a reward of +5.
* If the bird collides with the pipe or hits the ground, it will receive a penalty of -10.
* If the bird survives, it will be rewarded with +1 for each time step.

## Algorithm

* In this environment, the agent is trained using the Q-Learning Algorithm.

## Results

![Imgur](https://i.imgur.com/BE1O5Wa.png)
