import flappy_bird_gym
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
plt.style.use('default')


'''Q Learning Algorithm is used in this code'''

env = flappy_bird_gym.make('FlappyBird-v0') # Initializing the environment

Q_table = {}

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
graphs = {}
test_graphs = {}
def choose_action(Q_table,state_space,exp_prob):
    if np.random.uniform(0,1) < exp_prob and exp_prob > min_exp_prob:
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
    while not done and flag:
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
    exp_prob = (1-(epd+1)/num_epds)  # Linear decay of epsilon
    # exp_prob = np.exp((-(epd+1))*exp_decay)  # Exponential decay of epsilon
    if epd <= num_epds-1:
        if epd == 0:
            print("Training the agent......")
        rewards.append(R)
        epds.append(epd+1)
        returns.append(G)
        scores.append(score)
    else:
        if epd == num_epds:
            print("Testing the agent......")
        test_returns.append(G)
        test_epds.append(epd-num_epds)
        test_scores.append(score)
        test_rewards.append(R)
    if epd%1000==0:
        print("episode: ",epd)

graphs['reward'] = np.array(rewards)
graphs['Episode'] = np.array(epds)
graphs['return'] = np.array(returns)
graphs['score'] = np.array(scores)

test_graphs['test_reward'] = np.array(test_rewards)
test_graphs['test_Episode'] = np.array(test_epds)
test_graphs['test_return'] = np.array(test_returns)
test_graphs['test_score'] = np.array(test_scores)

df1 = pd.DataFrame(graphs)
df2 = pd.DataFrame(test_graphs)
df1.set_index('Episode',inplace=True)
df2.set_index('test_Episode',inplace=True)
sma_period1 = 1000
sma_period2 = 100
df1['SMA1'] = df1['return'].rolling(window=sma_period1).mean()
df1['SMA2'] = df1['reward'].rolling(window=sma_period1).mean()
df1['SMA3'] = df1['score'].rolling(window=sma_period1).mean()

df2['test_SMA1'] = df2['test_return'].rolling(window=sma_period2).mean()
df2['test_SMA2'] = df2['test_reward'].rolling(window=sma_period2).mean()
df2['test_SMA3'] = df2['test_score'].rolling(window=sma_period2).mean()
df1.dropna(inplace=True)
df2.dropna(inplace=True)
# SMA1 = list(df1['SMA1'])
# SMA2 = list(df1['SMA2'])
# SMA3 = list(df1['SMA3'])

'''Graphs for the trained episodes'''

'''Creating Subplots'''

fig1 , axes1 = plt.subplots(1,3,figsize = (14,7))
fig1.suptitle("FlappyBird-v0 using Q-Learning Algorithm for training",fontweight = 'bold',color = 'k')

axes1[0].set_title("Scores graph")
axes1[0].plot(df1[['score','SMA3']],label = ('SCORES','SMA3'))
axes1[0].set_xlabel("Number of Episodes")
# axes1[0].set_ylabel("Scores")
axes1[0].legend(loc = 0)
plt.grid()

axes1[1].set_title("Returns graph")
axes1[1].plot( df1[['return','SMA1']],label = ('RETURNS','SMA1'))
axes1[1].set_xlabel("Number of Episodes")
# axes1[1].set_ylabel("Returns")
axes1[1].legend(loc = 0)
plt.grid()

axes1[2].set_title("Rewards graph")
axes1[2].plot( df1[['reward','SMA2']],label =('REWARDS','SMA2'))
axes1[2].set_xlabel("Number of Episodes")
# axes1[2].set_ylabel("Rewards")
axes1[2].legend(loc = 0)
plt.grid()
plt.show()

'''Graphs for test episodes'''
fig2,axes2 = plt.subplots(1,3,figsize = (14,7))
fig2.suptitle("FlappyBird-v0 using Q-Learning Algorithm for testing",fontweight = 'bold',color = 'k')

axes2[0].set_title("Test scores graph")
axes2[0].plot(df2[['test_score','test_SMA3']],label = ('TEST_SCORES','TEST_SMA3'))
axes2[0].set_xlabel("Number of Episodes")
# axes2[0].set_ylabel("Test scores")
axes2[0].legend(loc = 0)
plt.grid()

axes2[1].set_title("Test returns graph")
axes2[1].plot( df2[['test_return','test_SMA1']],label = ('TEST_RETURNS','TEST_SMA1'))
axes2[1].set_xlabel("Number of Episodes")
# axes2[1].set_ylabel("Test returns")
axes2[1].legend(loc = 0)
plt.grid()

axes2[2].set_title("Test rewards graph")
axes2[2].plot( df2[['test_reward','test_SMA2']],label = ('TEST_REWARDS','TEST_SMA2'))
axes2[2].set_xlabel("Number of Episodes")
# axes2[2].set_ylabel("Test rewards")
axes2[2].legend(loc = 0)
plt.grid()
plt.show()




'''Rendering the flappy bird Environment by using the Q_Table'''

obs = env.reset()
score = 0
state_space = (obs[0],obs[1])
while True:
    env.render()
    if state_space in Q_table:
        action = np.argmax(Q_table[state_space])
    nxt_obs,reward,done,info = env.step(action)
    nxt_state_space = (nxt_obs[0],nxt_obs[1])
    state_space = nxt_state_space
    score = info["score"]
    time.sleep(1 / 30)  # Frames Per Second
    if done:
        print("Crashed!")
        break
print("Score: ",score)
env.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        