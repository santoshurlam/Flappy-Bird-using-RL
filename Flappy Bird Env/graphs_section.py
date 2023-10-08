
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''Unpickling the graphs'''

file1 = "g_values.pkl"
file1_obj = open(file1,'rb')
graphs = pickle.load(file1_obj)

'''Unpickling the test_graphs'''

file2 = "tg_values.pkl"
file2_obj = open(file2,'rb')
test_graphs = pickle.load(file2_obj)


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

SMA1 = list(df1['SMA1'])
SMA2 = list(df1['SMA2'])
SMA3 = list(df1['SMA3'])


'''Graphs for the trained episodes'''

'''Creating Subplots'''

fig1 , axes1 = plt.subplots(3,1,figsize = (12,10))
fig1.suptitle("FlappyBird-v0 using Q-Learning Algorithm for training",fontweight = 'bold',color = 'k')
plt.subplots_adjust(hspace=0.5)
axes1[0].set_title("Scores graph")
axes1[0].plot(df1[['score','SMA3']],label = ('SCORES','SMA3'))
axes1[0].set_xlabel("Number of Episodes")
axes1[0].set_ylabel("Scores")
axes1[0].legend(loc = 0)
axes1[0].grid()

axes1[1].set_title("Returns graph")
axes1[1].plot( df1[['return','SMA1']],label = ('RETURNS','SMA1'))
axes1[1].set_xlabel("Number of Episodes")
axes1[1].set_ylabel("Returns")
axes1[1].legend(loc = 0)
axes1[1].grid()

axes1[2].set_title("Rewards graph")
axes1[2].plot( df1[['reward','SMA2']],label =('REWARDS','SMA2'))
axes1[2].set_xlabel("Number of Episodes")
axes1[2].set_ylabel("Rewards")
axes1[2].legend(loc = 0)
axes1[2].grid()

# axes1[3].set_title("Time Steps graph")
# axes1[3].plot( df1[['time_steps','SMA4']],label =('TIME_STEPS','SMA4'))
# axes1[3].set_xlabel("Number of Episodes")
# axes1[2].set_ylabel("Time steps")
# axes1[3].legend(loc = 0)
# axes1[3].grid()

plt.show()

'''Graphs for test episodes'''

fig2,axes2 = plt.subplots(3,1,figsize = (12,10))
fig2.suptitle("FlappyBird-v0 using Q-Learning Algorithm for testing",fontweight = 'bold',color = 'k')
plt.subplots_adjust(hspace=0.5)

axes2[0].set_title("Test scores graph")
axes2[0].plot(df2[['test_SMA3']],label = ('TEST_SMA3'))
axes2[0].set_xlabel("Number of Episodes")
axes2[0].set_ylabel("Test scores")
axes2[0].legend(loc = 0)
axes2[0].grid()

axes2[1].set_title("Test returns graph")
axes2[1].plot( df2[['test_SMA1']],label = ('TEST_SMA1'))
axes2[1].set_xlabel("Number of Episodes")
axes2[1].set_ylabel("Test returns")
axes2[1].legend(loc = 0)
axes2[1].grid()

axes2[2].set_title("Test rewards graph")
axes2[2].plot( df2[['test_SMA2']],label = ('TEST_SMA2'))
axes2[2].set_xlabel("Number of Episodes")
axes2[2].set_ylabel("Test rewards")
axes2[2].legend(loc = 0)
axes2[2].grid()


# axes2[3].set_title("Test Time steps graph")
# axes2[3].plot( df2[['test_time_steps','test_SMA4']],label = ('TEST_TIME_STEPS','TEST_SMA4'))
# axes2[3].set_xlabel("Number of Episodes")
# axes2[2].set_ylabel("Test time steps")
# axes2[3].legend(loc = 0)
# axes2[3].grid()
plt.show()




        
        
        
        
        
        
        
        
        
        
        
        
        
        
