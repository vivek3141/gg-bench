import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to digits 1-9 (action 0 corresponds to digit 1, action 8 to digit 9)
        self.action_space = spaces.Discrete(9)
        # Observation is the cumulative number, an integer between 0 and 100 inclusive
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([100]), shape=(1,), dtype=int
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cumulative_number = 0
        self.current_player = 1  # Player 1 starts; can use 1 or -1 to represent players
        self.done = False

        observation = np.array([self.cumulative_number])
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            return (
                np.array([self.cumulative_number]),
                0,
                True,
                False,
                {},
            )

        # Map action (0-8) to digit (1-9)
        digit = action + 1
        new_number_str = f"{self.cumulative_number}{digit}"
        new_number = int(new_number_str)

        # Check if the move is valid
        if new_number > 100:
            # Invalid move; current player loses
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            info = {}
            observation = np.array([self.cumulative_number])
            return observation, reward, terminated, truncated, info

        # Update cumulative number
        self.cumulative_number = new_number

        # Check for victory condition
        if self.cumulative_number == 100:
            # Current player wins
            self.done = True
            reward = 1
            terminated = True
            truncated = False
            info = {}
            observation = np.array([self.cumulative_number])
            return observation, reward, terminated, truncated, info

        # Game continues; switch to other player
        self.current_player *= -1
        reward = 0
        terminated = False
        truncated = False
        info = {}
        observation = np.array([self.cumulative_number])
        return observation, reward, terminated, truncated, info

    def render(self):
        return f"Cumulative number is now: {self.cumulative_number}"

    def valid_moves(self):
        # Return a list of valid actions (indices of digits that can be appended without exceeding 100)
        valid_actions = []
        for action in range(9):  # Actions 0-8 correspond to digits 1-9
            digit = action + 1
            new_number_str = f"{self.cumulative_number}{digit}"
            new_number = int(new_number_str)
            if new_number <= 100:
                valid_actions.append(action)
        return valid_actions
