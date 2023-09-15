import matplotlib.pyplot as plt
from MonteCarlo import epds
from MonteCarlo import returns as R1
from MonteCarlo import steps as S1
from QLearning import episode
from QLearning import rewards as R2
from QLearning import steps_to_goal as S2
from SARSA import epds
from SARSA import returns as R3
from SARSA import steps_to_goal as S3
from SARSALambda import epds
from SARSALambda import returns as R4
from SARSALambda import steps_to_goal as S4

plt.plot(episode, R1, color='r', label='Monte-Carlo Algorithm')
plt.plot(episode, R2, color='g', label='Q-Learning Algorithm')
plt.plot(episode, R3, color='b', label='SARSA Algorithm')
plt.plot(episode, R4, color='m', label='SARSA Lambda Algorithm')
plt.xlabel("No. of Episodes")
plt.ylabel("Returns")
plt.title("MiniGrid-Empty-6x6-v0")
plt.legend()
plt.show()

plt.plot(episode, S1, color='r', label='Monte-Carlo Algorithm')
plt.plot(episode, S2, color='g', label='Q-Learning Algorithm')
plt.plot(episode, S3, color='b', label='SARSA Algorithm')
plt.plot(episode, S4, color='m', label='SARSA Lambda Algorithm')
plt.xlabel("No. of Episodes")
plt.ylabel("Steps to reach Goal")
plt.title("MiniGrid-Empty-6x6-v0")
plt.legend()
plt.show()




