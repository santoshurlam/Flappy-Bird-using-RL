import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-8x8-v0')

returns = []
epds = []
steps_to_goal = []
env.reset()
Q = {}
gamma = 0.99
alpha = 0.5
epsilon = 1.0
number_of_episodes = 60
for i in range(number_of_episodes+20):
    env.reset()
    if np.random.rand() < epsilon and epsilon > 0:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(Q[(1,1,0)])
    done = False
    Policy = []
    step = 1
    while not done:
        state = env.unwrapped.agent_pos
        direction = env.unwrapped.agent_dir
        state_space = (state[0], state[1], direction)
        if state_space not in Q:
            Q[state_space] = np.zeros(3)
        observation, reward, done, truncation, info = env.step(action)
        a = state_space
        b = action
        Policy.append(b)
        state = env.unwrapped.agent_pos
        direction = env.unwrapped.agent_dir
        state_space = (state[0], state[1], direction)
        if state_space not in Q:
            Q[state_space] = np.zeros(3)
        if np.random.rand() < epsilon and epsilon > 0:
            action = np.random.randint(0,3)
        else:
            action = np.argmax(Q[state_space])
        Q[a][b] += alpha*((reward + gamma * Q[state_space][action]) - Q[a][b])
        step+=1
    steps_to_goal.append(step)
    returns.append(reward)
    epds.append(i+1)
    epsilon = (-1*i/number_of_episodes) + 1
env = gym.make('MiniGrid-Empty-6x6-v0',render_mode = 'human')
# env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
n=0
while n < 1:
    env.reset()
    epd_rew = 0
    for i in range(len(Policy)):
        n_obs,rew,done,trunc,info= env.step(Policy[i])
        time.sleep(0.25)
        epd_rew += rew
        env.render()
    n += 1
env.close()
print("episode_reward: ",epd_rew)
print("optimal_policy: ",Policy)

"""Graphs between Number of episodes vs steps to reach goal and Number of episodes vs reward function."""

# plt.title("MiniGrid-Empty-8x8-v0 using SARSA Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using SARSA Algorithm")
plt.plot(epds,returns)
plt.xlabel("Number of episodes")
plt.ylabel("Reward at each episode")
plt.show()
# plt.title("MiniGrid-Empty-8x8-v0 using SARSA Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using SARSA Algorithm")
plt.plot(epds,steps_to_goal)
plt.xlabel("Number of episodes")
plt.ylabel("Steps to reach goal")
plt.show()
