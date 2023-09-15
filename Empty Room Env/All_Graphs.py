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




