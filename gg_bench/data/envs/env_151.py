import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Generate all possible valid equations and their required counts
        self.equations, self.equation_counts = self.generate_equations()
        self.action_space = spaces.Discrete(len(self.equations))

        # Observation space: counts of numbers 1 to 9 (each can be 0 to 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool with two of each number from 1 to 9
        self.number_pool_counts = np.array(
            [2] * 9, dtype=np.int8
        )  # Indices 0-8 correspond to numbers 1-9
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.number_pool_counts.copy(), {}  # Return observation and empty info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.number_pool_counts.copy(), -10, True, False, {}
        if action not in self.valid_moves():
            # Invalid move results in immediate loss
            self.done = True
            reward = -10
            return self.number_pool_counts.copy(), reward, True, False, {}
        # Valid move; update the number pool by removing used numbers
        counts = self.equation_counts[action]
        for num, count in counts.items():
            self.number_pool_counts[num - 1] -= count
        # Check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent cannot move; current player wins
            self.done = True
            reward = 1
            return self.number_pool_counts.copy(), reward, True, False, {}
        else:
            # Switch to the other player
            self.current_player *= -1
            reward = 0  # No immediate reward
            return (
                self.number_pool_counts.copy(),
                reward,
                False,
                False,
                {},
            )  # Continue game

    def render(self):
        # Generate a string representation of the current number pool
        pool_numbers = []
        for i in range(9):
            pool_numbers.extend([i + 1] * self.number_pool_counts[i])
        pool_str = "Number Pool: " + ", ".join(map(str, pool_numbers))
        return pool_str

    def valid_moves(self):
        if self.done:
            return []
        # Return a list of action indices corresponding to valid equations
        valid_moves = []
        for idx, counts in enumerate(self.equation_counts):
            is_valid = True
            for num, req_count in counts.items():
                if self.number_pool_counts[num - 1] < req_count:
                    is_valid = False
                    break
            if is_valid:
                valid_moves.append(idx)
        return valid_moves

    def generate_equations(self):
        equations = []
        equation_counts = []
        for A in range(1, 10):
            for B in range(1, 10):
                # Addition
                C = A + B
                if 1 <= C <= 9:
                    equations.append((A, "+", B, C))
                    counts = {}
                    for num in [A, B, C]:
                        counts[num] = counts.get(num, 0) + 1
                    equation_counts.append(counts)
                # Subtraction
                C = A - B
                if 1 <= C <= 9:
                    equations.append((A, "-", B, C))
                    counts = {}
                    for num in [A, B, C]:
                        counts[num] = counts.get(num, 0) + 1
                    equation_counts.append(counts)
                # Multiplication
                C = A * B
                if 1 <= C <= 9:
                    equations.append((A, "*", B, C))
                    counts = {}
                    for num in [A, B, C]:
                        counts[num] = counts.get(num, 0) + 1
                    equation_counts.append(counts)
                # Division
                if B != 0 and A % B == 0:
                    C = A // B
                    if 1 <= C <= 9:
                        equations.append((A, "/", B, C))
                        counts = {}
                        for num in [A, B, C]:
                            counts[num] = counts.get(num, 0) + 1
                        equation_counts.append(counts)
        return equations, equation_counts
