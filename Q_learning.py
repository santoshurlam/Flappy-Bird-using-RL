import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
# import useful_funcs as uf
# env = gym.make("MiniGrid-Empty-6x6-v0")
env = gym.make('MiniGrid-Empty-8x8-v0')
Q_table = {}
alpha = 0.1
exp_prob = 1.0
min_exp_prob = 0.0
num_epds = 100
max_steps = 300
gamma = 0.99
returns = []
epds = []
steps_to_goal = []
def choose_action(Q_table,state_space,exp_prob):
    if np.random.uniform(0,1) < exp_prob and exp_prob > min_exp_prob:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(Q_table[state_space])
    return action
def Q_updation(Q_table,state_space,action,nxt_state_space):
    old_Q = Q_table[state_space][action]
    max_new_Q = np.max([Q_table[nxt_state_space][a] for a in range(3)])
    new_Q = (1-alpha)*old_Q + alpha*(reward + gamma*max_new_Q)
    old_Q = new_Q
    return old_Q

for epd in range(num_epds+20):
    env.reset()
    state = env.unwrapped.agent_pos
    dirn = env.unwrapped.agent_dir
    x,y = state[0],state[1]
    state_space = (x,y,dirn)
    done = False
    trunc = False
    steps = 1
    policy = []
    G = 0
    while not done and not trunc:
        if state_space not in Q_table:
            Q_table[state_space] = np.zeros(3,dtype=float)
        steps += 1
        action = choose_action(Q_table,state_space,exp_prob)
        policy.append(action)
        nxt_obs,reward,done,trunc,info = env.step(action)
        G = reward + gamma*G
        next_state = env.unwrapped.agent_pos
        n_dirn = env.unwrapped.agent_dir
        x,y = next_state[0],next_state[1]
        nxt_state_space = (x,y,n_dirn)
        if nxt_state_space not in Q_table:
            Q_table[nxt_state_space] = np.zeros(3,dtype=float)
        Q_table[state_space][action] = Q_updation(Q_table,state_space,action,nxt_state_space)
        state_space = nxt_state_space
    exp_prob = (1-(epd+1)/num_epds)
    epds.append(epd+1)
    returns.append(G)
    steps_to_goal.append(steps)
print("policy: ",policy)      
print(Q_table)
# env = gym.make("MiniGrid-Empty-6x6-v0",render_mode = 'human')
env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
env.reset()
epd_rew = 0
for i in range(len(policy)):
    n_obs,rew,done,trunc,info= env.step(policy[i])
    time.sleep(0.25)
    epd_rew += rew
    env.render()
env.close()

# plt.title("MiniGrid-Empty-6x6-v0 using Q-Learning Algorithm")
plt.title("MiniGrid-Empty-8x8-v0 using Q-Learning Algorithm")

plt.plot(epds,steps_to_goal)
plt.xlabel("Number of episodes")
plt.ylabel("Steps to reach goal")
plt.show()

plt.title("MiniGrid-Empty-8x8-v0 using Q-Learning Algorithm")
# plt.title("MiniGrid-Empty-6x6-v0 using Q-Learning Algorithm")
plt.plot(epds, returns)
plt.xlabel("Number of episodes")
plt.ylabel("Reward at each episode")
plt.show()