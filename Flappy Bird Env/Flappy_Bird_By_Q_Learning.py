import flappy_bird_gym
import numpy as np
import matplotlib.pyplot as plt
import time

'''Q Learning Algorithm is used in this code'''

env = flappy_bird_gym.make('FlappyBird-v0') # Initializing the environment

Q_table = {}

'''Assigning the Constants'''

alpha = 0.4
exp_prob = 1.0
min_exp_prob = 0.0
exp_decay = 0.01
num_epds = 5000
max_steps = 3000
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

for epd in range(num_epds+1000):
    state = env.reset()
    score = 0
    x,y = state[0],state[1]
    state_space = (x,y)
    done = False
    steps = 1
    policy = []
    G = 1
    R = 1
    while not done:
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
        test_epds.append(epd+1)
        test_scores.append(score)
        test_rewards.append(R)
    if epd%1000==0:
        print("episode: ",epd)
   

'''Graphs for the trained episodes'''

plt.title("FlappyBird-v0 using Q-Learning Algorithm")
plt.plot(epds,scores)
plt.xlabel("Number of episodes")
plt.ylabel("Score at each episode")
plt.show()

plt.title("FlappyBird-v0 using Q-Learning Algorithm")
plt.plot(epds, returns)
plt.xlabel("Number of episodes")
plt.ylabel("Return at each episode")
plt.show()

plt.title("FlappyBird-v0 using Q-Learning Algorithm")
plt.plot(epds, rewards)
plt.xlabel("Number of episodes")
plt.ylabel("Total reward at each episode")
plt.show()


plt.title("All algorithms graphs")
plt.plot(epds,scores, color = 'r',label = 'SCORES')
plt.plot(epds,returns, color = 'g', label = 'RETURNS')
plt.plot(epds,rewards, color = 'b',label ='REWARDS')
plt.xlabel("Number of Episodes")
plt.ylabel("Returns and Scores and Rewards")
plt.show()

'''Graphs for test episodes'''

plt.title("FlappyBird-v0 using Q-Learning Algorithm for test episodes")
plt.plot(test_epds,test_scores)
plt.xlabel("Number of episodes")
plt.ylabel("Score at each episode")
plt.show()

plt.title("FlappyBird-v0 using Q-Learning Algorithm for test episodes")
plt.plot(test_epds, test_returns)
plt.xlabel("Number of episodes")
plt.ylabel("Reward at each episode")
plt.show()

plt.title("FlappyBird-v0 using Q-Learning Algorithm")
plt.plot(test_epds, test_rewards)
plt.xlabel("Number of episodes")
plt.ylabel("Total reward at each episode")
plt.show()

plt.title("All algorithms graphs for test episodes")
plt.plot(test_epds,test_scores, color = 'r',label = 'SCORES')
plt.plot(test_epds,test_returns, color = 'g', label = 'RETURNS')
plt.plot(test_epds,test_rewards, color = 'b',label ='REWARDS')
plt.xlabel("Number of Episodes")
plt.ylabel("Returns and Scores and Rewards")
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        