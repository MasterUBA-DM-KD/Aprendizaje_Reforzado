import numpy as np

# Initialize the bandit, for a=1 to k, Q(a) = 0, N(a) = 0
# Loop:
# 1. With probability epsilon select a random action
# 2. Otherwise, select a = argmax Q(a)
# 3. Take action a, observe R(a) (Reward has random mean and constant standard deviation)
# 4. N(a) = N(a) + 1
# 5. Q(a) = Q(a) + (1/N(a))[R(a) - Q(a)]


class EpsilonGreedyBandit:
    """
    Epsilon-Greedy Bandit algorithm, a simple exploration-exploitation strategy.

    Parameters
    ----------
    n_arms : int
        Number of arms (actions)
    epsilon : float
        Probability of selecting a random arm (exploration)
    """

    def __init__(self, n_arms: int, epsilon: float) -> None:
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.rewards_in_time = [0.0]
        self.arm_in_time = []

    def select_arm(self) -> int:
        """
        Select an arm (action) to play based on the epsilon-greedy policy.

        Returns
        -------
        int
            Index of the selected arm (action)
        """
        if np.random.rand() > self.epsilon:
            # Exploitation: select the arm with the highest estimated value
            arm = int(np.argmax(self.values))
            self.arm_in_time.append(arm)
            return arm
        else:
            # Exploration: select a random arm
            arm = np.random.randint(0, self.n_arms)
            self.arm_in_time.append(arm)
            return arm

    def update(self, arm: int, reward: float) -> None:
        """
        Update the value estimates for the selected arm based on the observed reward.

        Parameters
        ----------
        arm: int
            Index of the selected arm
        reward: float
            Observed reward for the selected arm
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = value + (1 / n) * (reward - value)
        self.values[arm] = new_value
        self.rewards_in_time.append(reward)

    def reset(self) -> None:
        """Reset the bandit to the initial state."""
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    @property
    def estimated_values(self) -> np.ndarray:
        """
        Estimated values of the arms (actions).

        Returns
        -------
        np.ndarray
            Array with the estimated value of each arm.
        """
        return self.values

    @property
    def selected_arm_counts(self) -> np.ndarray:
        """
        Number of times each arm has been selected.

        Returns
        -------
        np.ndarray
            Array with the number of times each arm has been selected.
        """
        return self.counts

    @property
    def rewards_arm(self) -> list[float]:
        """
        Rewards obtained for each arm in time.

        Returns
        -------
        list[float]
            Array with the rewards obtained in each time step.
        """
        return self.rewards_in_time

    @property
    def selected_arms(self) -> list[int]:
        """
        Arm selected in each time step.

        Returns
        -------
        list[int]
            List with the arm selected in each time step.
        """
        return self.arm_in_time
