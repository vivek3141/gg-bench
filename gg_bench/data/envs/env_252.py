import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Bit Flip Duel Environment.

    Two players take turns flipping bits in their own sequences.
    Flipping a bit also flips the corresponding bit in the opponent's sequence.
    The goal is to be the first to turn all bits in your own sequence to 1.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 5 bits to choose from (positions 0 to 4)
        self.action_space = spaces.Discrete(5)
        # Observation space consists of both players' sequences (10 bits in total)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Each player's sequence starts with 5 zeros
        self.sequence_player1 = np.zeros(5, dtype=np.int8)
        self.sequence_player2 = np.zeros(5, dtype=np.int8)
        self.current_player = 1  # Player 1 starts the game
        self.done = False  # Game is not over
        self.info = {}  # Additional info

        return self._get_observation(), self.info  # Observation and info

    def step(self, action):
        # Validate action
        if self.done or action not in self.valid_moves():
            # Invalid move
            return (
                self._get_observation(),
                -10,  # Penalty for invalid move
                True,  # Game ends
                False,
                self.info,
            )

        # Flip the selected bit in both players' sequences
        self.sequence_player1[action] = 1 - self.sequence_player1[action]
        self.sequence_player2[action] = 1 - self.sequence_player2[action]

        # Check if the current player has won
        if self.current_player == 1:
            player_sequence = self.sequence_player1
        else:
            player_sequence = self.sequence_player2

        if np.all(player_sequence == 1):
            # Current player wins
            self.done = True
            return (
                self._get_observation(),
                1,  # Reward for winning
                True,  # Game ends
                False,
                self.info,
            )

        # Switch to the other player
        self.current_player *= -1

        # Continue the game
        return self._get_observation(), 0, False, False, self.info

    def render(self):
        # Return a string representation of the game state
        board_str = f"Player 1 Sequence: {' '.join(map(str, self.sequence_player1))}\n"
        board_str += f"Player 2 Sequence: {' '.join(map(str, self.sequence_player2))}\n"
        board_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return board_str

    def valid_moves(self):
        # All moves are valid (positions 0 to 4) if the game is not over
        if not self.done:
            return list(range(5))
        else:
            return []

    def _get_observation(self):
        # Combine both sequences into a single observation
        return np.concatenate([self.sequence_player1, self.sequence_player2])
