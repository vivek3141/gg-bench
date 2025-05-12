import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define target number
        self.target_number = 100

        # Define action space: Multipliers 2 to 9 represented by indices 0 to 7
        self.action_space = spaces.Discrete(8)

        # Define observation space
        # Observation consists of:
        # - current_number normalized between 0 and 1
        # - multipliers_available: 1 if available, 0 if used
        # Observation shape: (1 current_number + 8 multipliers) = (9,)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.multipliers_available = [1] * 8  # 1 if available, 0 if used
        self.done = False
        self.current_player = 1  # Player 1 starts

        observation = self._get_observation()
        return observation, {}  # Observation and info

    def _get_observation(self):
        # Normalize current number between 0 and 1
        max_number = self.target_number * 10  # Assume numbers won't exceed this
        normalized_current_number = min(self.current_number / max_number, 1.0)
        # Construct observation array
        observation = np.array(
            [normalized_current_number] + self.multipliers_available, dtype=np.float32
        )
        return observation

    def step(self, action):
        # Check if the action is valid
        if action < 0 or action >= 8:
            # Invalid action: out of bounds
            return self._get_observation(), -10, True, False, {}
        if self.multipliers_available[action] == 0:
            # Invalid action: multiplier already used
            return self._get_observation(), -10, True, False, {}

        if self.done:
            return self._get_observation(), 0, True, False, {}

        multiplier = action + 2  # Map action index to multiplier

        # Update the current number
        self.current_number *= multiplier

        # Mark the multiplier as used
        self.multipliers_available[action] = 0

        # Check for win condition
        if self.current_number >= self.target_number:
            self.done = True
            reward = 1  # Win
            return self._get_observation(), reward, True, False, {}

        # Check if no multipliers are left
        if sum(self.multipliers_available) == 0:
            self.done = True
            reward = 0  # Game over without winning
            return self._get_observation(), reward, True, False, {}

        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Return the observation, reward, done, truncated, and info
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        multipliers_list = [
            str(i + 2)
            for i, avail in enumerate(self.multipliers_available)
            if avail == 1
        ]
        s = (
            f"Current Number: {self.current_number}\n"
            f"Available Multipliers: {', '.join(multipliers_list)}\n"
            f"Current Player: Player {self.current_player}"
        )
        return s

    def valid_moves(self):
        # Return a list of valid action indices
        return [i for i, avail in enumerate(self.multipliers_available) if avail == 1]
