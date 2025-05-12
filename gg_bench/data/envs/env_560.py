import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the action space: numbers 1 to 20 (0 to 19 indices)
        self.action_space = spaces.Discrete(20)

        # Define the observation space:
        # - First 20 elements: used numbers (-1: player 2, 0: unused, 1: player 1)
        # - Next element: last_number (0 if none, else 1-20)
        # - Next element: current player (-1 or 1)
        self.observation_space = spaces.Box(
            low=np.array([-1] * 20 + [0] + [-1], dtype=np.int32),
            high=np.array([1] * 20 + [20] + [1], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.used_numbers = np.zeros(20, dtype=np.int32)
        self.last_number = 0  # No last number at the start
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Observation includes used numbers, last number, and current player
        observation = np.concatenate(
            (
                self.used_numbers,
                np.array([self.last_number, self.current_player], dtype=np.int32),
            )
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            observation = np.concatenate(
                (
                    self.used_numbers,
                    np.array([self.last_number, self.current_player], dtype=np.int32),
                )
            )
            return observation, 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = np.concatenate(
                (
                    self.used_numbers,
                    np.array([self.last_number, self.current_player], dtype=np.int32),
                )
            )
            return observation, reward, True, False, {}

        # Valid move
        self.used_numbers[action] = self.current_player
        self.last_number = action + 1  # Numbers are from 1 to 20
        self.current_player *= -1  # Switch player

        # Check if opponent has valid moves
        if not self.valid_moves():
            # Opponent cannot move; current player wins
            self.done = True
            reward = 1
            observation = np.concatenate(
                (
                    self.used_numbers,
                    np.array([self.last_number, self.current_player], dtype=np.int32),
                )
            )
            return observation, reward, True, False, {}
        else:
            # Game continues
            reward = 0
            observation = np.concatenate(
                (
                    self.used_numbers,
                    np.array([self.last_number, self.current_player], dtype=np.int32),
                )
            )
            return observation, reward, False, False, {}

    def render(self):
        # Visualization of the current game state
        used_numbers_display = [
            "_" if x == 0 else ("X" if x == 1 else "O") for x in self.used_numbers
        ]
        used_numbers_str = " ".join(
            f"{i+1}:{used_numbers_display[i]}" for i in range(20)
        )
        seq_str = (
            f"Last number: {self.last_number if self.last_number != 0 else 'None'}"
        )
        player_str = (
            f"Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        return f"{used_numbers_str}\n{seq_str}\n{player_str}"

    def valid_moves(self):
        valid_actions = []
        if self.last_number == 0:
            # First turn; any unused number is valid
            valid_actions = [i for i in range(20) if self.used_numbers[i] == 0]
        else:
            for i in range(20):
                n = i + 1  # Actual number (1 to 20)
                if self.used_numbers[i] == 0:
                    if self.last_number % n == 0 or n % self.last_number == 0:
                        valid_actions.append(i)
        return valid_actions
