import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the action space: numbers 1-9 correspond to actions 0-8
        self.action_space = spaces.Discrete(9)

        # Define the observation space: an array of 9 elements
        # 0: Number is in the pool (available)
        # 1: Number selected by Player 1
        # 2: Number selected by Player 2
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int8)

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(9, dtype=np.int8)  # All numbers are in the pool initially
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.state.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return current state
            return self.state.copy(), 0, self.done, False, {}

        if action < 0 or action >= 9 or self.state[action] != 0:
            # Invalid move: number not in the pool
            self.done = True
            return self.state.copy(), -10, True, False, {}

        # Valid move: update the state
        self.state[action] = self.current_player

        # Check for a winning condition
        player_numbers = [
            i + 1 for i in range(9) if self.state[i] == self.current_player
        ]
        if len(player_numbers) >= 3:
            # Check all combinations of 3 numbers
            for combo in itertools.combinations(player_numbers, 3):
                if self.is_arithmetic_sequence(combo):
                    # Current player wins
                    self.done = True
                    return self.state.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if all numbers are selected
        if np.all(self.state != 0):
            # Game continues until a player wins
            pass

        return self.state.copy(), 0, self.done, False, {}

    def render(self):
        number_pool = [str(i + 1) for i in range(9) if self.state[i] == 0]
        player1_collection = [str(i + 1) for i in range(9) if self.state[i] == 1]
        player2_collection = [str(i + 1) for i in range(9) if self.state[i] == 2]
        output = f"Number Pool: {' '.join(number_pool)}\n"
        output += f"Player 1 Collection: {' '.join(player1_collection)}\n"
        output += f"Player 2 Collection: {' '.join(player2_collection)}\n"
        return output

    def valid_moves(self):
        return [i for i in range(9) if self.state[i] == 0]

    def is_arithmetic_sequence(self, combo):
        # Check all permutations of the combination for arithmetic sequence
        for perm in itertools.permutations(combo):
            diff1 = perm[1] - perm[0]
            diff2 = perm[2] - perm[1]
            if diff1 == diff2:
                return True
        return False
