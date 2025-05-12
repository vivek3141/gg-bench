import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=8):
        super(CustomEnv, self).__init__()

        # Length of the binary number
        self.N = N

        # Define action space: Actions from 0 to (2N - 2)
        # Actions 0 to N-1: Flip single bit at position 0 to N-1
        # Actions N to 2N - 2: Flip two adjacent bits starting at position 0 to N-2
        self.action_space = spaces.Discrete(2 * self.N - 1)

        # Define observation space: Binary number with N bits
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.N,), dtype=int)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the binary number to all ones
        self.binary_number = np.ones(self.N, dtype=int)
        # Player 1 starts first (can be 1 or -1 to represent players, but not essential here)
        self.current_player = 1
        # Game is not done yet
        self.done = False
        # No info to return
        info = {}
        return self.binary_number.copy(), info  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            reward = -10  # Penalty for acting when game is over
            return self.binary_number.copy(), reward, True, False, {}

        # Decode the action
        if action < self.N:
            # Flip single bit at position action
            pos = action
            if self.binary_number[pos] == 1:
                # Valid move
                self.binary_number[pos] = 0
                # Check for win
                if np.all(self.binary_number == 0):
                    self.done = True
                    reward = 1  # Win reward
                    return self.binary_number.copy(), reward, True, False, {}
                else:
                    # Valid move, but game continues
                    reward = -10  # Penalty per valid move
                    self.current_player *= -1  # Switch player
                    return self.binary_number.copy(), reward, False, False, {}
            else:
                # Invalid move (bit already 0)
                self.done = True
                reward = -100  # Heavy penalty for invalid move
                return self.binary_number.copy(), reward, True, False, {}
        elif action < 2 * self.N - 1:
            # Flip two adjacent bits starting at position pos
            pos = action - self.N
            if pos < self.N - 1:
                if self.binary_number[pos] == 1 and self.binary_number[pos + 1] == 1:
                    # Valid move
                    self.binary_number[pos] = 0
                    self.binary_number[pos + 1] = 0
                    # Check for win
                    if np.all(self.binary_number == 0):
                        self.done = True
                        reward = 1  # Win reward
                        return self.binary_number.copy(), reward, True, False, {}
                    else:
                        # Valid move, but game continues
                        reward = -10  # Penalty per valid move
                        self.current_player *= -1  # Switch player
                        return self.binary_number.copy(), reward, False, False, {}
                else:
                    # Invalid move (bits not both 1 or not adjacent)
                    self.done = True
                    reward = -100  # Heavy penalty for invalid move
                    return self.binary_number.copy(), reward, True, False, {}
            else:
                # Invalid action (position out of range)
                self.done = True
                reward = -100  # Heavy penalty for invalid action
                return self.binary_number.copy(), reward, True, False, {}
        else:
            # Invalid action (out of action space)
            self.done = True
            reward = -100  # Heavy penalty for invalid action
            return self.binary_number.copy(), reward, True, False, {}

    def render(self):
        binary_str = " ".join(str(bit) for bit in self.binary_number)
        return f"Current Binary Number: {binary_str}"

    def valid_moves(self):
        valid_actions = []
        # Check for single bit flips
        for i in range(self.N):
            if self.binary_number[i] == 1:
                valid_actions.append(i)
        # Check for flipping two adjacent bits
        for i in range(self.N - 1):
            if self.binary_number[i] == 1 and self.binary_number[i + 1] == 1:
                valid_actions.append(self.N + i)
        return valid_actions
