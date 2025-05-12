import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers 1-9 (indices 0-8)
        self.action_space = spaces.Discrete(9)
        # Observation space: state of numbers 1-9 (-1: Player 2, 0: Available, 1: Player 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers = np.zeros(
            9, dtype=np.int8
        )  # 0: Available, 1: Player 1, -1: Player 2
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.numbers.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.numbers.copy(), 0, True, False, {}

        if self.numbers[action] != 0:
            # Invalid move
            self.done = True
            return self.numbers.copy(), -10, True, False, {}

        # Valid move
        self.numbers[action] = self.current_player

        # Check for victory
        player_numbers = [
            i + 1 for i, val in enumerate(self.numbers) if val == self.current_player
        ]
        if self.check_victory(player_numbers):
            self.done = True
            return self.numbers.copy(), 1, True, False, {}

        # Check if game is over (no more moves)
        if np.all(self.numbers != 0):
            self.done = True
            return self.numbers.copy(), 0, True, False, {}

        # Switch player
        self.current_player *= -1
        return self.numbers.copy(), 0, False, False, {}

    def check_victory(self, player_numbers):
        if len(player_numbers) < 3:
            return False

        from itertools import combinations, permutations

        # Generate all combinations of 3 numbers
        for combo in combinations(player_numbers, 3):
            # Generate all permutations of the combination
            for perm in permutations(combo):
                a, b, c = perm
                if a + b == c or a - b == c:
                    return True
        return False

    def render(self):
        available_numbers = [
            str(i + 1) for i, val in enumerate(self.numbers) if val == 0
        ]
        player1_numbers = [str(i + 1) for i, val in enumerate(self.numbers) if val == 1]
        player2_numbers = [
            str(i + 1) for i, val in enumerate(self.numbers) if val == -1
        ]

        state_str = (
            f"Available Numbers: {' '.join(available_numbers)}\n"
            f"Player 1 Hand: {' '.join(player1_numbers)}\n"
            f"Player 2 Hand: {' '.join(player2_numbers)}\n"
        )

        return state_str

    def valid_moves(self):
        return [i for i, val in enumerate(self.numbers) if val == 0]
