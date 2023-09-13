import  gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
env = gym.make("MiniGrid-Empty-6x6-v0")
# env = gym.make('MiniGrid-Empty-8x8-v0')

returns = []
epds = []
steps_to_goal = []
env.reset()
Q = {}
gamma = 0.99
alpha = 0.1
epsilon = 1.0
lambda_value = 0.9
number_of_episodes = 150
max_steps = 200
for epd in range(number_of_episodes+20):
    env.reset()
    in_state = env.unwrapped.agent_pos
    in_dirn = env.unwrapped.agent_dir
    x,y = in_state[0],in_state[1]
    in_state_space = (x,y,in_dirn)
    if np.random.rand() < epsilon and epsilon > 0:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(Q[in_state_space])
    done = False
    trunc = False
    Policy = []
    E_Traces = {}
    step = 1
    while not done and step < max_steps:
        state = env.unwrapped.agent_pos
        direction = env.unwrapped.agent_dir
        state_space = (state[0], state[1], direction)
        if state_space not in Q:
            Q[state_space] = np.zeros(3)
        if state_space not in E_Traces:
            E_Traces[state_space] = np.zeros(3)
            
        observation, reward, done, truncation, info = env.step(action)
        Policy.append(action)
        n_state = env.unwrapped.agent_pos
        n_dirn = env.unwrapped.agent_dir
        nxt_state_space = (n_state[0], n_state[1], n_dirn)
        if nxt_state_space not in Q:
            Q[nxt_state_space] = np.zeros(3)
        if nxt_state_space not in E_Traces:
            E_Traces[nxt_state_space] = np.zeros(3)
        if np.random.rand() < epsilon and epsilon > 0:
            nxt_action = np.random.randint(0,3)
        else:
            nxt_action = np.argmax(Q[nxt_state_space])
        td_error = reward + gamma*Q[nxt_state_space][nxt_action] - Q[state_space][action]
        E_Traces[state_space][action] += 1
        Q[state_space] += alpha*td_error*E_Traces[state_space]
        E_Traces[state_space] = lambda_value*gamma*E_Traces[state_space]
        step+=1
        action = nxt_action
        # state_space = nxt_state_space
    steps_to_goal.append(step)
    returns.append(reward)
    epds.append(epd+1)
    epsilon = (-1*(epd)/number_of_episodes) + 1
    # print("No. of episodes Completed: ", epd+1)
    # print("Eligibility Traces : ",E_Traces)
    # print("Policy : ",Policy)
    # print("\n")
# print(Q)
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

# plt.title("MiniGrid-Empty-8x8-v0 using SARSA Lambda Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using SARSA Lambda Algorithm")
plt.plot(epds,returns)
plt.xlabel("Number of episodes")
plt.ylabel("Reward at each episode")
plt.show()
# plt.title("MiniGrid-Empty-8x8-v0 using SARSA Lambda Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using SARSA Lambda Algorithm")
plt.plot(epds,steps_to_goal)
plt.xlabel("Number of episodes")
plt.ylabel("Steps to reach goal")
plt.show()