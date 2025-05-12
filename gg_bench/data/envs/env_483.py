import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: integers from 0 to 8 (representing numbers 1 to 9)
        self.action_space = spaces.Discrete(9)

        # Observation space: array of 9 integers, each can be -1, 0, or 1
        # -1: selected by player 2, 0: available, 1: selected by player 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Winning combinations: list of sets of numbers that form valid arithmetic progressions
        self.winning_combinations = [
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7},
            {5, 6, 7, 8},
            {6, 7, 8, 9},
            {1, 3, 5, 7},
            {2, 4, 6, 8},
            {3, 5, 7, 9},
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 0: available, 1: selected by player 1, -1: selected by player 2
        self.available_numbers = np.zeros(9, dtype=np.int8)
        self.player_sequences = {1: [], -1: []}  # Sequences of both players
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return self.available_numbers.copy()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if the action is valid
        if action < 0 or action >= 9 or self.available_numbers[action] != 0:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move: update state
        number_picked = action + 1  # Numbers range from 1 to 9
        self.available_numbers[action] = self.current_player
        self.player_sequences[self.current_player].append(number_picked)

        # Check for a win condition
        player_numbers = self.player_sequences[self.current_player]
        if len(player_numbers) >= 4:
            # Check all combinations of 4 numbers
            for combo in combinations(player_numbers, 4):
                if set(combo) in self.winning_combinations:
                    self.done = True
                    return self._get_obs(), 1, True, False, {}

        # Check if all numbers have been selected without a winner
        if np.all(self.available_numbers != 0):
            self.done = True
            return self._get_obs(), 0, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return self._get_obs(), 0, False, False, {}

    def render(self):
        output = "Available Numbers: "
        for i in range(9):
            if self.available_numbers[i] == 0:
                output += f"{i+1} "
        output += "\n"
        output += f"Player 1's Sequence: {sorted(self.player_sequences[1])}\n"
        output += f"Player 2's Sequence: {sorted(self.player_sequences[-1])}\n"
        return output

    def valid_moves(self):
        return [i for i in range(9) if self.available_numbers[i] == 0]
