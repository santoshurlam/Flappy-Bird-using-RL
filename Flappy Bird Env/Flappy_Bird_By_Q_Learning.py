import flappy_bird_gym
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import pickle

'''Q Learning Algorithm is used in this code'''

env = flappy_bird_gym.make('FlappyBird-v0') # Initializing the environment

Q_table = {}
graphs = {}
test_graphs = {}

'''Assigning the Constants'''

alpha = 0.4
exp_prob = 1.0
min_exp_prob = 0.0
exp_decay = 0.01
num_epds = 500000
max_steps = 10000
gamma = 0.99

'''Creating the Empty lists'''

returns = []
epds = []
scores = []
rewards = []
test_returns = []
test_epds = []
test_scores = []
test_rewards = []
time_steps = []
test_time_steps = []

def choose_action(Q_table,state_space,exp_prob):
    if np.random.uniform(0,1) < exp_prob:
        action = np.random.randint(0,2)
    else:
        action = np.argmax(Q_table[state_space])
    return action
def Q_updation(Q_table,state_space,action,nxt_state_space):
    old_Q = Q_table[state_space][action]
    max_new_Q = np.max([Q_table[nxt_state_space][a] for a in range(2)])
    new_Q = (1-alpha)*old_Q + alpha*(reward + gamma*max_new_Q)
    old_Q = new_Q
    return old_Q

for epd in range(num_epds+5000):
    state = env.reset()
    score = 0
    x,y = state[0],state[1]
    state_space = (x,y)
    done = False
    steps = 1
    flag = True
    G = 1
    R = 1
    while not done and flag :
        if epd >= 200000:
            if steps > max_steps:
                flag = False
        if state_space not in Q_table:
            Q_table[state_space] = np.zeros(2,dtype=float)
        steps += 1
        action = choose_action(Q_table,state_space,exp_prob)
        nxt_obs,reward,done,info = env.step(action)
        if info['score'] > score:
            reward += 5
        score = info['score']
        G = reward + gamma*G
        R += reward
        next_state = nxt_obs
        x,y = next_state[0],next_state[1]
        nxt_state_space = (x,y)
        if nxt_state_space not in Q_table:
            Q_table[nxt_state_space] = np.zeros(2,dtype=float)
        Q_table[state_space][action] = Q_updation(Q_table,state_space,action,nxt_state_space)
        state_space = nxt_state_space
    exp_prob = (1-(epd+1)/(num_epds+5000))  # Linear decay of epsilon
    # exp_prob = np.exp((-(epd+1))*exp_decay)  # Exponential decay of epsilon
    if epd <= num_epds-1:
        if epd == 0:
            print("Training the agent......")
        rewards.append(R)
        epds.append(epd+1)
        returns.append(G)
        scores.append(score)
        time_steps.append(steps)
    else:
        if epd == num_epds:
            print("Testing the agent......")
        test_returns.append(G)
        test_epds.append(epd-num_epds)
        test_scores.append(score)
        test_rewards.append(R)
        test_time_steps.append(steps)
    if epd%1000==0:
        print("episode: ",epd)


'''Pickling the Q_Table'''

Q_file = "Q_values.pkl"
Q_file_obj = open(Q_file,'wb')
pickle.dump(Q_table,Q_file_obj)
Q_file_obj.close()


graphs['reward'] = np.array(rewards)
graphs['Episode'] = np.array(epds)
graphs['return'] = np.array(returns)
graphs['score'] = np.array(scores)

'''Pickling the graphs'''

graphs_file = "g_values.pkl"
g_file_obj = open(graphs_file,'wb')
pickle.dump(graphs,g_file_obj)
g_file_obj.close()


test_graphs['test_reward'] = np.array(test_rewards)
test_graphs['test_Episode'] = np.array(test_epds)
test_graphs['test_return'] = np.array(test_returns)
test_graphs['test_score'] = np.array(test_scores)

'''Pickling the test_graphs'''

test_graphs_file = "tg_values.pkl"
tg_file_obj = open(test_graphs_file,'wb')
pickle.dump(test_graphs,tg_file_obj)
tg_file_obj.close()



print("Max Training Score: ",max(scores))
print("Max Testing Score: ",max(test_scores))

