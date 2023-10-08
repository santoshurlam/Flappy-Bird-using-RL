import flappy_bird_gym
import pickle
import time
import numpy as np

'''Unpickling the Q_Table'''

file3 = "Q_values.pkl"
file3_obj = open(file3,'rb')
Q_table = pickle.load(file3_obj)


'''Rendering the flappy bird Environment by using the Q_Table'''

env = flappy_bird_gym.make("FlappyBird-v0")

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