
# Frozen Lake Environment

**Description:**

* This Environment structure is of by 4x4 and 8x8 Grids and those each grid may contain a frozen lake or hole and the final goal state has gift.
* It is model-based Environment.
**Aim:**
* The agent has to reach the goal state in optimal path by using dynamic programming.

## State Space
* Each grid or cell or state represnted by integer starting from 0 to 15 for 4x4 grid and 0 to 63 for 8x8 grid.
* If the agent moves towards boundary of grid from a state by taking an action it will be in the same state.
## Action Space

There are 4 actions for every state that agent can take ;
  - Left -  0
  - Down -  1
  - Right - 2
  - Up -    3

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
# Flappy Bird Environment

**Description:**

* This Environment consists of agent (Bird) and also randomly generating pipes (Upper and Lower pipes) with constant pipe gap size (gap between upper and lower pipe) and Base & Background. The bird (agent) is can move in vertical direction only and the pipes which are generating along with Background moving horizontal direction.

* Flappy bird Environment is a model-free Environment.

**Aim:**
* The bird (agent) has to learn to score itself by crossing the pipes using Q-Learning Algorithm.

![Flappy Bird Render](https://i.imgur.com/epEFm8u.gif)


## Requirements
* Matplotlib
* NumPy
* flappy_bird_gym (Cloned Repository from [Flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym))
**Note:** Algorithm file was created in cloned Repository folder to directly import the flappy_bird_gym into the code file.

```bash
  pip install Matplotlib
  pip install NumPy
```
    
## Environment Variables

**State Space**

* This Environment consists of state as location of the bird (agent) center.
* Location is the coordinates of the bird which briefs about the horizontal distance between the bird's center to next pipe's center and vertical distance between bird's center to hole center of lower pipe.
* State always getting reset after either hitting the pipe or crashing on base.
* Agent moves upward direction only if it flaps since the PLAYER_VEL_ROT = 0 and player_rot = 0 degrees but initially it was 45 degrees.


