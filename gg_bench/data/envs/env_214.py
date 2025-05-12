import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(15), actions from 0 to 14
        self.action_space = spaces.Discrete(15)

        # Observation space: MultiBinary(5), the binary string of 5 bits
        self.observation_space = spaces.MultiBinary(5)

        # Winning patterns for Player 1 and Player 2
        self.player1_winning_pattern = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        self.player2_winning_pattern = np.array([0, 1, 0, 1, 0], dtype=np.int8)

        # Map action indices to flip positions
        self.action_to_flip = self._generate_action_mapping()

        # Initialize the game state
        self.reset()

    def _generate_action_mapping(self):
        # Generate a mapping from action indices to bits to flip
        mapping = {}
        action_index = 0

        # Actions for flipping one bit (positions 1 to 5)
        for pos in range(5):
            mapping[action_index] = [pos]
            action_index += 1

        # Actions for flipping two bits (all combinations of positions 1 to 5)
        for i in range(5):
            for j in range(i + 1, 5):
                mapping[action_index] = [i, j]
                action_index += 1

        return mapping

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the binary string to all zeros
        self.binary_string = np.zeros(5, dtype=np.int8)
        self.current_player = 1  # Player 1 starts the game
        self.done = False
        return self.binary_string.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if the action is valid
        if not self.action_space.contains(action):
            # Invalid action
            return self.binary_string.copy(), -10, True, False, {}

        # Flip the bits according to the action
        flip_positions = self.action_to_flip[action]
        self.binary_string[flip_positions] = 1 - self.binary_string[flip_positions]

        # Check for win or loss conditions
        if self.current_player == 1:
            if np.array_equal(self.binary_string, self.player1_winning_pattern):
                # Player 1 wins
                reward = 1
                self.done = True
            elif np.array_equal(self.binary_string, self.player2_winning_pattern):
                # Player 1 loses by creating Player 2's pattern
                reward = -10
                self.done = True
            else:
                # Game continues
                reward = 0
        elif self.current_player == 2:
            if np.array_equal(self.binary_string, self.player2_winning_pattern):
                # Player 2 wins
                reward = 1
                self.done = True
            elif np.array_equal(self.binary_string, self.player1_winning_pattern):
                # Player 2 loses by creating Player 1's pattern
                reward = -10
                self.done = True
            else:
                # Game continues
                reward = 0
        else:
            # Invalid player (should not happen)
            raise Exception("Invalid current_player")

        # Switch to the next player if the game is not over
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        # Return observation, reward, done, truncated, and info
        return self.binary_string.copy(), reward, self.done, False, {}

    def render(self):
        # Return a visual representation of the game state as a string
        positions = "Positions:  [1] [2] [3] [4] [5]\n"
        bit_values = "Bit Values:  " + "   ".join(map(str, self.binary_string))
        return positions + bit_values

    def valid_moves(self):
        # Return a list of valid action indices (0 to 14)
        return list(range(15))
