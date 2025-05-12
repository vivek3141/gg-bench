import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 8 possible actions corresponding to bit positions 1-8 (0-indexed)
        self.action_space = spaces.Discrete(8)

        # Observation space: Flattened array of both players' bits (2 players x 8 bits)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Each player starts with bits set to all ones
        self.player_bits = np.ones((2, 8), dtype=np.int32)
        # Player 1 starts first (index 0)
        self.current_player = 0
        self.done = False
        self.winner = None
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten the player bits array to a 16-element vector
        return self.player_bits.flatten()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if action is valid
        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Process the attack
        attacker = self.current_player
        opponent = 1 - self.current_player
        bit_pos = action  # Bit positions are 0-indexed (0-7)

        # Attacker's bit at position becomes 0 (expended)
        self.player_bits[attacker, bit_pos] = 0

        # Opponent's bit at position
        if self.player_bits[opponent, bit_pos] == 1:
            # Opponent's bit is captured and set to 0
            self.player_bits[opponent, bit_pos] = 0
            # Check for win condition
            if np.all(self.player_bits[opponent] == 0):
                # Attacker wins
                self.done = True
                self.winner = attacker
                return self._get_obs(), 1, True, False, {}

        # Switch to the other player
        self.current_player = opponent

        # Reward is -10 for a valid move
        return self._get_obs(), -10, False, False, {}

    def render(self):
        # Build the string representation
        player1_bits = " ".join(map(str, self.player_bits[0]))
        player2_bits = " ".join(map(str, self.player_bits[1]))
        s = f"Player 1's bits: {player1_bits}\n"
        s += f"Player 2's bits: {player2_bits}\n"
        s += f"Current player: Player {self.current_player + 1}"
        return s

    def valid_moves(self):
        # Returns a list of valid actions (bit positions 0-7) for the current player
        return [i for i in range(8) if self.player_bits[self.current_player, i] == 1]
