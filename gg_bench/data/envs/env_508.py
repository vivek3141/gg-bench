import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - append 'A', 1 - append 'B'
        self.action_space = spaces.Discrete(2)
        # Observation: Array of size 10, each element can be 0 (empty), 1 ('A'), or 2 ('B')
        self.observation_space = spaces.Box(low=0, high=2, shape=(10,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.string = []  # Shared string starts empty
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game is not over

        # Return initial observation and info
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Default values
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions or self.done:
            # Invalid action or game already over
            reward = -10  # Penalty for invalid move
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Append the chosen character to the shared string
        char = "A" if action == 0 else "B"
        self.string.append(char)

        # Check for win condition
        current_string = "".join(self.string)
        if "ABBA" in current_string:
            # Current player wins
            reward = 1
            terminated = True
            self.done = True
        elif len(self.string) == 10:
            # String reached maximum length without 'ABBA'; current player loses
            reward = -1
            terminated = True
            self.done = True
        else:
            # Game continues; switch to the other player
            self.current_player = 2 if self.current_player == 1 else 1

        # Return observation and game info
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        # Return the current state of the shared string
        return f"Shared String: {''.join(self.string)}"

    def valid_moves(self):
        # Moves are valid if the game is not over and the string length is less than 10
        if self.done or len(self.string) >= 10:
            return []  # No valid moves
        else:
            return [0, 1]  # Indices for actions: append 'A' or 'B'

    def _get_observation(self):
        # Convert the shared string into an observation array
        obs = np.zeros(10, dtype=np.int32)
        char_to_int = {"A": 1, "B": 2}
        for idx, char in enumerate(self.string):
            obs[idx] = char_to_int[char]
        return obs
