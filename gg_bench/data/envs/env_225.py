import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space: 5 locks * 2 increment options = 10 actions
        self.action_space = spaces.Discrete(10)

        # Observation space: For each lock, current_value and unlock_value
        # Unlock values range from 3 to 5
        # Current values range from 0 to 5
        self.observation_space = spaces.Box(
            low=np.array([0] * 5 + [3] * 5),
            high=np.array([5] * 5 + [5] * 5),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Assign random unlock values between 3 and 5 to each lock
        self.unlock_values = np.random.randint(3, 6, size=5)

        # Initialize current values to 0
        self.current_values = np.zeros(5, dtype=np.int32)

        # Initialize locked status for each lock
        self.locked = np.ones(5, dtype=bool)  # True for Locked, False for Unlocked

        # Initialize unlocked count
        self.unlocked_count = 0

        # First player
        self.current_player = 1  # 1 for Player 1, -1 for Player 2

        # Game done flag
        self.done = False

        # Prepare initial observation
        observation = np.concatenate([self.current_values, self.unlock_values])

        return observation, {}

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_obs(), 0, True, False, {}

        # Map action to lock_index and increment_value
        lock_index = action // 2
        increment_value = 1 + (action % 2)

        # Check if lock is locked
        if not self.locked[lock_index]:
            # Invalid move: lock already unlocked
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Get remaining to unlock
        remaining_to_unlock = (
            self.unlock_values[lock_index] - self.current_values[lock_index]
        )

        # Check if increment is allowed
        if increment_value > remaining_to_unlock:
            # Invalid move: increment exceeds unlock value
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move
        # Increment the lock's current value
        self.current_values[lock_index] += increment_value

        # Check if lock is now unlocked
        if self.current_values[lock_index] == self.unlock_values[lock_index]:
            self.locked[lock_index] = False
            self.unlocked_count += 1

            # Check if game is over
            if self.unlocked_count == 5:
                # Current player wins
                self.done = True
                return self._get_obs(), 1, True, False, {}

        # Switch player
        self.current_player *= -1

        return self._get_obs(), 0, False, False, {}

    def render(self):
        output = "Current Lock Status:\n"
        for i in range(5):
            status = "Unlocked" if not self.locked[i] else "Locked"
            output += f"Lock {i+1}: {self.current_values[i]} / {self.unlock_values[i]} ({status})\n"
        return output

    def valid_moves(self):
        valid_actions = []
        for action in range(10):
            lock_index = action // 2
            increment_value = 1 + (action % 2)
            if self.locked[lock_index]:
                remaining_to_unlock = (
                    self.unlock_values[lock_index] - self.current_values[lock_index]
                )
                if increment_value <= remaining_to_unlock:
                    valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        return np.concatenate([self.current_values, self.unlock_values])
