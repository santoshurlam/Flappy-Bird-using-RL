import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-8x8-v0')
env.reset()
Q_table= {}


def Q_table_generator():
    for epd in range(20):
        # env.reset()
        done = False
        while not done:
            state = env.unwrapped.agent_pos
            dirn = env.unwrapped.agent_dir
            a = np.random.randint(0, 3)
            n_obs, rew, done, trunc, info = env.step(a)
            ss = (state[0], state[1], dirn)
            Q_table[ss] = np.zeros(3, dtype=float)
    my_keys = list(Q_table.keys())
    my_keys.sort()
    sorted_dict = {i: Q_table[i] for i in my_keys}
    return sorted_dict


Q_table = Q_table_generator()
# policy = uf.random_policy_generator()
# Q_table = {}
steps_to_goal = []
# epsilons = []
returns = []
epds = []
num_eps = 150
gamma = 0.9
alpha = 0.4
epsilon = 1.0
final_epsilon = 0.0


def policy_updation(Q_table, state_space, epsilon):
    if np.random.uniform(0,1) < epsilon and epsilon > final_epsilon:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(Q_table[state_space])
    return action


def Q_values(episode):
    G = 0
    for i in reversed(range(len(episode))):
        state_space, action, reward = episode[i]
        G = reward + G*gamma
        Q_table[state_space][action] += alpha *(G - Q_table[state_space][action])
    returns.append(G)
    return Q_table


for epd in range(num_eps+20):
    state = env.reset()
    done = False
    trunc = False
    episode = []
    policy = []
    max_steps = 600
    steps = 1
    # print("epd_no: ",epd+1)
    while not done and not trunc:
        steps += 1
        dirn = env.unwrapped.agent_dir
        state = env.unwrapped.agent_pos
        x, y = state[0], state[1]
        state_space = (x, y, dirn)
        # if state_space not in Q_table:
        #     Q_table[state_space] = np.zeros(3, dtype=float)
        action = policy_updation(Q_table, state_space, epsilon)
        policy.append(action)
        n_obs, reward, done, trunc, info = env.step(action)
        episode.append((state_space, action, reward))
        dirn = env.unwrapped.agent_dir
        state = env.unwrapped.agent_pos
        x, y = state[0], state[1]
        state_space = (x, y, dirn)
    steps_to_goal.append(steps)
    epds.append(epd+1)
    Q_values(episode)
    epsilon = (1-(epd+1)/num_eps)

my_keys = list(Q_table.keys())
my_keys.sort()
sorted_dict = {i: Q_table[i] for i in my_keys}
print(policy)
# print("Q_table\n",sorted_dict,end="\n")


# env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
env = gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')
env.reset()
epd_rew = 0
for i in range(len(policy)):
    n_obs, rew, done, trunc, info = env.step(policy[i])
    time.sleep(0.25)
    epd_rew += rew
    env.render()
env.close()

"""Graphs between Number of episodes vs steps to reach goal and Number of episodes vs reward function."""

# plt.title("MiniGrid-Empty-8x8-v0 using Monte Carlo Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using Monte Carlo Algorithm")
plt.plot(epds, returns)
plt.xlabel("Number of episodes")
plt.ylabel("Reward at each episode")
plt.show()
# plt.title("MiniGrid-Empty-8x8-v0 using Monte Carlo Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using Monte Carlo Algorithm")
plt.plot(epds, steps_to_goal)
plt.xlabel("Number of episodes")
plt.ylabel("Steps to reach goal")
plt.show()

