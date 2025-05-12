import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - add symbol to left end, 1 - add symbol to right end
        self.action_space = spaces.Discrete(2)
        # Observations: Array of size 11, values can be -1, 0 or 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []  # Start with empty sequence
        self.current_player = 1  # Player 1 starts with 'X' (represented by 1)
        self.done = False  # Game is not over yet
        self.last_player = None  # Keep track of last player who made a move
        self.observation = np.zeros(
            11, dtype=np.int32
        )  # The observation is a fixed length array

        return self.observation, {}  # Return initial observation and info

    def step(self, action):
        if self.done:
            # The game is over, no further actions can be taken
            return self.observation.copy(), -10, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return self.observation.copy(), -10, True, False, {}

        # Map action to side
        side = "left" if action == 0 else "right"

        # Add the current player's symbol to the appropriate end
        symbol = self.current_player  # 1 for 'X', -1 for 'O'
        if side == "left":
            self.sequence.insert(0, symbol)
        else:
            self.sequence.append(symbol)

        # Update the observation array
        seq_len = len(self.sequence)
        self.observation[:seq_len] = self.sequence
        self.observation[seq_len:] = 0  # Remaining positions are zeros

        # Check for palindromes
        palindrome_detected = self.check_for_palindromes()

        if palindrome_detected:
            # Current player loses
            self.done = True
            reward = 0  # No reward
            return self.observation.copy(), reward, True, False, {}
        elif seq_len == 11:
            # Sequence is full, current player wins
            self.done = True
            reward = 1  # Current player wins
            return self.observation.copy(), reward, True, False, {}
        else:
            # Game continues, switch players
            self.last_player = self.current_player
            self.current_player *= -1  # Switch player
            reward = 0  # Normal move reward
            return self.observation.copy(), reward, False, False, {}

    def check_for_palindromes(self):
        seq_len = len(self.sequence)
        for length in range(
            3, seq_len + 1
        ):  # Palindromes of length 3 up to current sequence length
            for start_idx in range(seq_len - length + 1):
                subseq = self.sequence[start_idx : start_idx + length]
                if subseq == subseq[::-1]:
                    # Palindrome detected
                    return True
        return False

    def render(self):
        symbol_map = {1: "X", -1: "O", 0: " "}
        seq_str = " ".join([symbol_map[s] for s in self.sequence])
        return f"Sequence: {seq_str}"

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1]
