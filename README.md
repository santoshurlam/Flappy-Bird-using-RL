
# Frozen Lake Environment

**Description:**

* This Environment structure is of by 4x4 and 8x8 Grids and those each grid may contain a frozen lake or hole and the final goal state has gift.
* It is model-based Environment.

**Aim:**
* The agent has to reach the goal state in optimal path by using dynamic programming.


### Deterministic Environment

![](https://i.imgur.com/RlJjiZM.gif) ![](https://i.imgur.com/1dpekVN.gif)

### Stochastic Environment

![](https://i.imgur.com/9dF44vt.gif)


## State Space
* Each grid or cell or state represnted by integer starting from 0 to 15 for 4x4 grid and 0 to 63 for 8x8 grid.
* If the agent moves towards boundary of grid from a state by taking an action it will be in the same state.

## Action Space

The actions can be taken by agent in any state are;

```bash
  Left - 0
  Down - 1
  Right- 2
  Up   - 3
```

## Reward
* If the agent falls in hole then reward is 0 and if it landed on frozen lake then also reward is 0 but if the agent reaches the goal state it receives the reward of 1.

## Algorithms
In this Dynamic Programming method is used for convergence of policy.
This can also be done by two ways
### Policy Iteration
  - Evaluating Value function for all states
  - Acting greedily towards policy using action value function evaluated using value function
  - Iterate it until the convergence of policy.
### Value Iteration
  - Method for finding the optimal value function by updating the bellman equation iteratively.
  - Taking an action greedily from all actions for a particular state by using action value function.
  - Iterate it until the convergence of policy.

# Minigrid Environment

**Description:**

* This Minigrid Environment is Empty room, it consists of one agent and one goal state and it doesn't contain any obstacles.
*  This have MiniGrid-Empty-6x6-v0 and MiniGrid-Empty-8x8-v0 environments.
* This is a model-free Environment.

**Aim:**

* The agent has to reach the goal state in most optimal way.

![](https://i.imgur.com/4lCwL8g.gif) ![](https://i.imgur.com/tIZ0FNG.gif)

## State Space

- There are 16 states in MiniGrid-Empty-6x6-v0 environment and each state is represented by (x,y) where x = 1 to 4 and y = 1 to 4 
- And there are 36 states in MiniGrid-Empty-8x8-v0 environment and each cell is represented by (x,y) where x = 1 to 6 and y = 1 to 6
- State space also contain the direction of the agent at that state, the direction are as follows,
  	- 0 - Right 
  	- 1 - Down
  	- 2 - Left
  	- 3 - Up
- Obseravtion contain iamge array which can be used to identify where the agent is in environment.


## Action Space

There are three action agent can take to change state or direction;

```bash
  - 0 - Turn Left
  - 1 - Turn Right
  - 2 - Move Forward

```

## Rewards

* Every state has reward 0 except at goal state.
* Goal state has reward 1.

## Algorithms
```bash
 Monte-Carlo
 SARSA
 SARSA Lambda
 Q-Learning
```
## Results
#### MiniGrid-Empty-6x6-v0
![Graph 1](https://i.imgur.com/cISSqmA.png)

![Graph 2](https://i.imgur.com/TbHxtFL.png)
# Flappy Bird Environment

**Description:**

* This Environment consists of agent (Bird) and also randomly generating pipes (Upper and Lower pipes) with constant pipe gap size (gap between upper and lower pipe) and Base & Background. The bird (agent) is can move in vertical direction only and the pipes which are generating along with Background moving horizontal direction.

* Flappy bird Environment is a model-free Environment.

**Aim:**
* The bird (agent) has to learn to score itself by crossing the pipes using Q-Learning Algorithm.

<p align = "center">
    <img src = "https://i.imgur.com/ZgW3wYP.gif" alt = "Flappy bird">
</p>

## Requirements
* Matplotlib
* NumPy
* flappy_bird_gym (Cloned Repository from [Flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym))

**Note:** Algorithm file was created in cloned Repository folder to directly import the flappy_bird_gym into the code file.

```bash
  pip install Matplotlib
  pip install NumPy
```
    
## State Space

* This Environment consists of state as location of the bird (agent) center.
* Location is the coordinates of the bird which briefs about the horizontal distance between the bird's center to next pipe's center and vertical distance between bird's center to hole center of lower pipe.
* State always getting reset after either hitting the pipe or crashing on base.
* Agent moves upward direction only if it flaps since the PLAYER_VEL_ROT = 0 and player_rot = 0 degrees but initially it was 45 degrees.

## Action Space

The actions can be taken by the agent in any state are;

```bash
  Flap to fly- 1
  Do nothing - 0
```
## Reward 

* If the bird crosses the pipes it receives +5 as reward.
* If the bird hits the pipe or falls on ground it receives -10 as reward.
* If the bird stays alive it receives the +1 as reward for every time step.

## Algorithm

* Q-Learning Algorithm is used to train the agent in this environment.

## Results

![Imgur](https://i.imgur.com/BE1O5Wa.png)
