**STEP ONE : FROZEN LAKE OPENAI GYM**

**Description**: For better understanding of the **Policy** and **Value** Iteration using the Frozen lake environment for both Deterministic and Stochastic of fully observable environments.

![Frozen Lake Gym](https://www.gymlibrary.dev/_images/frozen_lake.gif)

**IMPLEMENTATION**

**ACTION SPACE**

The action space consists of 4 actions -

	LEFT - 0
	DOWN - 1
	RIGHT- 2
	UP   - 3

**STATE SPACE**

* For 4x4 grid there are 16 cells and each cell represents a integer starting from 0 to 15.
* Any cell may contain a obstacle (Hole) or Frozen lake and the aim of the agent is to reach the Goal in optimal way using policy and value iteration.

**REWARD FUNCTION**

* +1 if the agent reaches the goal cell.
* 0 otherwise.

**ALGORITHM**
* Policy and Value iteration algorithms are used in this Environment to get the optimal policy.

**RESULTS**

![Frozen Lake result](https://github.com/santoshurlam/Summer_Project26/assets/99114485/a1c50257-0204-4192-8d92-920ae44618e8)
