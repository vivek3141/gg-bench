import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # There are 8 possible actions:
        # Actions 0-3: Flip own bit at positions 0-3
        # Actions 4-7: Flip opponent's bit at positions 0-3
        self.action_space = spaces.Discrete(8)
        # The observation will be a 12-dimensional vector:
        # [target_bits (4), player1_bits (4), player2_bits (4)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.int8)
        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly select a target binary number of 4 bits
        self.target = np.random.randint(0, 2, 4, dtype=np.int8)
        # Initialize both players' binaries to 0000
        self.player1_binary = np.zeros(4, dtype=np.int8)
        self.player2_binary = np.zeros(4, dtype=np.int8)
        # Game state variables
        self.done = False
        self.current_player = 1  # Player 1 starts
        return self._get_obs(), {}

    def _get_obs(self):
        # Concatenate target, player1's binary, and player2's binary into a single observation
        return np.concatenate([self.target, self.player1_binary, self.player2_binary])

    def step(self, action):
        # Check if the game has already ended
        if self.done:
            return self._get_obs(), -10, True, False, {}
        # Check if the action is valid
        if action < 0 or action >= 8:
            self.done = True
            return self._get_obs(), -10, True, False, {}
        # Determine own and opponent binaries based on current player
        if self.current_player == 1:
            own_binary = self.player1_binary
            opponent_binary = self.player2_binary
        else:
            own_binary = self.player2_binary
            opponent_binary = self.player1_binary
        # Perform the action
        if 0 <= action <= 3:
            # Flip a bit in own binary
            bit_position = action
            own_binary[bit_position] = 1 - own_binary[bit_position]
        elif 4 <= action <= 7:
            # Flip a bit in opponent's binary
            bit_position = action - 4
            opponent_binary[bit_position] = 1 - opponent_binary[bit_position]
        else:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}
        # Check for win condition
        if np.array_equal(own_binary, self.target):
            self.done = True
            return self._get_obs(), 1, True, False, {}
        # Switch to the other player
        self.current_player = 1 if self.current_player == 2 else 2
        return self._get_obs(), 0, False, False, {}

    def render(self):
        # Create a string representation of the game state
        target_str = "".join(map(str, self.target))
        player1_str = "".join(map(str, self.player1_binary))
        player2_str = "".join(map(str, self.player2_binary))
        output = f"Target Binary Number: {target_str}\n"
        output += f"Player 1's Binary: {player1_str}\n"
        output += f"Player 2's Binary: {player2_str}\n"
        output += f"Current Player: Player {self.current_player}\n"
        return output

    def valid_moves(self):
        # All actions from 0 to 7 are valid in this game
        return list(range(8))
