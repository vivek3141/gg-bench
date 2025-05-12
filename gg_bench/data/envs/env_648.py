import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(20) for numbers 1 to 20 (indices 0 to 19)
        self.action_space = spaces.Discrete(20)

        # Observation space: Array of 21 integers
        # Entries 0 to 19: numbers available (1) or not (0)
        # Entry 20: Parity requirement (-1, 0, 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(21,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers are initially available
        self.available_numbers = np.ones(20, dtype=np.int8)
        self.parity_required = 0  # 0 means any parity (start of the game)
        self.last_number = None
        self.done = False
        # Create the observation
        observation = np.append(self.available_numbers, self.parity_required)
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if action is valid
        valid_moves = self.valid_moves()
        if action not in valid_moves or self.done:
            # Invalid move or game is over
            self.done = True
            observation = np.append(self.available_numbers, self.parity_required)
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        # Remove the number from available numbers
        self.available_numbers[action] = 0
        number_picked = action + 1  # Numbers are from 1 to 20

        # Update parity requirement for next player
        if number_picked % 2 == 0:
            self.parity_required = -1  # Next player must pick odd number
        else:
            self.parity_required = 1  # Next player must pick even number

        # Check if opponent has any valid moves
        opponent_valid_moves = self.valid_moves()

        if not opponent_valid_moves:
            # Opponent has no valid moves, current player wins
            self.done = True
            observation = np.append(self.available_numbers, self.parity_required)
            return (
                observation,
                1,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Game continues
        observation = np.append(self.available_numbers, self.parity_required)
        return (
            observation,
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        available_numbers = [i + 1 for i in range(20) if self.available_numbers[i] == 1]
        parity_str = (
            "Any parity"
            if self.parity_required == 0
            else (
                "Even number required"
                if self.parity_required == 1
                else "Odd number required"
            )
        )
        return f"Available numbers: {available_numbers}\nNext move: {parity_str}\n"

    def valid_moves(self):
        # Returns a list of valid actions (indices) based on available numbers and parity required
        valid_actions = []
        for i in range(20):
            if self.available_numbers[i] == 1:
                number = i + 1
                if self.parity_required == 0:
                    # Any parity is acceptable
                    valid_actions.append(i)
                elif self.parity_required == 1 and number % 2 == 0:
                    # Need to pick even number
                    valid_actions.append(i)
                elif self.parity_required == -1 and number % 2 == 1:
                    # Need to pick odd number
                    valid_actions.append(i)
        return valid_actions
