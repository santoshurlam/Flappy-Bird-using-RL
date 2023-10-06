import matplotlib.pyplot as plt

from monte_carlo import returns as R1
from monte_carlo import steps_to_goal as S1

from Q_learning import returns as R2
from Q_learning import steps_to_goal as S2

from sarsa import returns as R3
from sarsa import steps_to_goal as S3

from sarsa_Lambda import returns as R4
from sarsa_Lambda import steps_to_goal as S4

episode = [epd for epd in range(200)]

fig1,axes1 = plt.subplots(2,2,figsize = (12,8))
fig1.suptitle("Returns Plot of MiniGrid-Empty-6x6-v0",fontweight = 'bold',color = 'k')
plt.subplots_adjust(hspace=0.3)

axes1[0,0].plot(episode, R1, label='Monte-Carlo Algorithm')
axes1[0,0].set_title("Monte-Carlo Algorithm")
axes1[0,1].plot(episode, R2, label='Q-Learning Algorithm')
axes1[0,1].set_title("Q-Learning Algorithm")
axes1[1,0].plot(episode, R3, label='SARSA Algorithm')
axes1[1,0].set_title("SARSA Algorithm")
axes1[1,1].plot(episode, R4, label='SARSA Lambda Algorithm')
axes1[1,1].set_title("SARSA Lambda Algorithm")

axes1[0,0].grid()
axes1[0,1].grid()
axes1[1,0].grid()
axes1[1,1].grid()

axes1[0,0].set_xlabel("No. of Episodes")
axes1[0,1].set_xlabel("No. of Episodes")
axes1[1,0].set_xlabel("No. of Episodes")
axes1[1,1].set_xlabel("No. of Episodes")

axes1[0,0].set_ylabel("Returns")
axes1[0,1].set_ylabel("Returns")
axes1[1,0].set_ylabel("Returns")
axes1[1,1].set_ylabel("Returns")

plt.show()

fig2,axes2 = plt.subplots(2,2,figsize = (12,8))
fig2.suptitle("Steps to Goal plot of MiniGrid-Empty-6x6-v0",fontweight = 'bold',color = 'k')
plt.subplots_adjust(hspace=0.3)

axes2[0,0].plot(episode, S1, label='Monte-Carlo Algorithm')
axes2[0,0].set_title("Monte-Carlo Algorithm")
axes2[0,1].plot(episode, S2, label='Q-Learning Algorithm')
axes2[0,1].set_title("Q-Learning Algorithm")
axes2[1,0].plot(episode, S3, label='SARSA Algorithm')
axes2[1,0].set_title("SARSA Algorithm")
axes2[1,1].plot(episode, S4, label='SARSA Lambda Algorithm')
axes2[1,1].set_title("SARSA Lambda Algorithm")

axes2[0,0].grid()
axes2[0,1].grid()
axes2[1,0].grid()
axes2[1,1].grid()

axes2[0,0].set_xlabel("No. of Episodes")
axes2[0,1].set_xlabel("No. of Episodes")
axes2[1,0].set_xlabel("No. of Episodes")
axes2[1,1].set_xlabel("No. of Episodes")

axes2[0,0].set_ylabel("Steps to Goal")
axes2[0,1].set_ylabel("Steps to Goal")
axes2[1,0].set_ylabel("Steps to Goal")
axes2[1,1].set_ylabel("Steps to Goal")

plt.show()



