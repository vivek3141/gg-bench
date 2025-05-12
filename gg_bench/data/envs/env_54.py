import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=21, max_sequence_length=100):
        super(CustomEnv, self).__init__()

        self.N = N  # Target number N
        self.max_sequence_length = max_sequence_length

        # Define action space: possible sums from 0 to N-1
        self.action_space = spaces.Discrete(self.N)

        # Define observation space: the Fibonacci sequence padded to max_sequence_length
        self.observation_space = spaces.Box(
            low=0, high=self.N, shape=(self.max_sequence_length,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = [1, 1]  # Start with [1, 1]
        self.last_opponent_move = None  # No move made by the opponent yet
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Initialize the padded sequence
        self.padded_sequence = np.zeros(self.max_sequence_length, dtype=np.int32)
        self.padded_sequence[: len(self.sequence)] = self.sequence

        info = {}
        return self.padded_sequence.copy(), info  # Observation and info

    def valid_moves(self):
        valid_sums = set()
        sequence_len = len(self.sequence)

        # Generate all possible sums of adjacent numbers
        for i in range(sequence_len - 1):
            sum_pair = self.sequence[i] + self.sequence[i + 1]

            # Apply game rules
            if sum_pair >= self.N:
                continue
            if sum_pair == self.last_opponent_move:
                continue
            if sum_pair == self.sequence[-1]:
                continue

            valid_sums.add(sum_pair)
        return list(valid_sums)

    def step(self, action):
        if self.done:
            return self.padded_sequence.copy(), -10, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move, current player loses
            self.done = True
            return self.padded_sequence.copy(), -10, True, False, {}

        # Valid move, append action to the sequence
        self.sequence.append(action)
        self.padded_sequence[: len(self.sequence)] = self.sequence
        self.last_player_move = action

        # Switch to the next player
        self.current_player *= -1
        self.last_opponent_move = action  # For the next player's move

        # Check if the opponent has valid moves
        opponent_valid_moves = self.valid_moves()

        if not opponent_valid_moves:
            # Opponent cannot make a move, current player wins
            self.done = True
            return self.padded_sequence.copy(), 1, True, False, {}

        # Game continues
        return self.padded_sequence.copy(), -10, False, False, {}

    def render(self):
        sequence_str = "Sequence: " + str(self.sequence)
        return sequence_str

    def valid_moves(self):
        valid_sums = set()
        sequence_len = len(self.sequence)

        for i in range(sequence_len - 1):
            sum_pair = self.sequence[i] + self.sequence[i + 1]

            if sum_pair >= self.N:
                continue
            if sum_pair == self.last_opponent_move:
                continue
            if sum_pair == self.sequence[-1]:
                continue

            valid_sums.add(sum_pair)
        return list(valid_sums)
