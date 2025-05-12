import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 6 possible actions
        self.action_space = spaces.Discrete(6)

        # Observation space: An array of 6 numbers from 1 to 5
        self.observation_space = spaces.Box(low=1, high=5, shape=(6,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the codes for both players
        self.player_codes = {
            1: np.random.randint(1, 6, size=3),  # Numbers from 1 to 5
            2: np.random.randint(1, 6, size=3),
        }

        # Current player: 1 or 2
        self.current_player = 1

        # Game over flag
        self.done = False

        # Return the initial observation and info
        return self._get_observation(), {}

    def step(self, action):
        if self.done or not self.action_space.contains(action):
            # Invalid action or game over
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Perform the action
        reward = 0
        self._perform_action(action)

        # Check if the current player has won
        if np.all(np.diff(self.player_codes[self.current_player]) >= 0):
            # Current player wins
            reward = 1
            self.done = True
        else:
            # Switch to the next player
            self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Return the observation, reward, done, truncated, and info
        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        s = f"Player {self.current_player}'s turn\n"
        s += f"Your code: {self.player_codes[self.current_player]}\n"
        s += f"Opponent's code: {self.player_codes[3 - self.current_player]}\n"
        return s

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(6))

    def _get_observation(self):
        # Observation is the current player's code followed by the opponent's code
        return np.concatenate(
            (
                self.player_codes[self.current_player],
                self.player_codes[3 - self.current_player],
            )
        )

    def _perform_action(self, action):
        if action == 0:
            # Swap positions 1 and 2 in own code
            self.player_codes[self.current_player][[0, 1]] = self.player_codes[
                self.current_player
            ][[1, 0]]
        elif action == 1:
            # Swap positions 1 and 3 in own code
            self.player_codes[self.current_player][[0, 2]] = self.player_codes[
                self.current_player
            ][[2, 0]]
        elif action == 2:
            # Swap positions 2 and 3 in own code
            self.player_codes[self.current_player][[1, 2]] = self.player_codes[
                self.current_player
            ][[2, 1]]
        elif 3 <= action <= 5:
            # Swap with opponent's code at the same position
            pos = action - 3  # Positions are 0-based
            temp = self.player_codes[self.current_player][pos]
            self.player_codes[self.current_player][pos] = self.player_codes[
                3 - self.current_player
            ][pos]
            self.player_codes[3 - self.current_player][pos] = temp
        # No else clause needed since action validity is checked in step()
