import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 18 actions (numbers 1-9, each can be added or subtracted)
        self.action_space = spaces.Discrete(18)

        # Define observation space:
        # - First 9 elements represent the availability of numbers 1-9 (1 if available, 0 if used)
        # - Last element is the running total, ranging from -50 to 50
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [-50], dtype=np.float32),
            high=np.array([1] * 9 + [50], dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            9, dtype=np.float32
        )  # Numbers 1-9 are available
        self.running_total = 0  # Running total starts at zero
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        return self._get_obs(), self.info  # Return observation and info

    def _get_obs(self):
        # Combine the available numbers and running total into a single observation
        return np.concatenate((self.available_numbers, [self.running_total])).astype(
            np.float32
        )

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, False, self.info

        # Map action to number and operation
        number_index = action // 2  # Index from 0 to 8
        operation = action % 2  # 0 for add, 1 for subtract

        # Check if the selected number is available
        if self.available_numbers[number_index] == 0:
            # Invalid move: number has already been used
            self.done = True
            reward = -10
            return self._get_obs(), reward, self.done, False, self.info

        # Retrieve the actual number
        number = number_index + 1  # Numbers 1 to 9

        # Perform the operation
        if operation == 0:
            # Add the number
            self.running_total += number
        else:
            # Subtract the number
            self.running_total -= number

        # Mark the number as used
        self.available_numbers[number_index] = 0

        # Check for losing condition
        if self.running_total == 0:
            # Current player loses
            self.done = True
            reward = -1  # Negative reward for losing
            return self._get_obs(), reward, self.done, False, self.info

        # Check if there are any numbers left to play
        if not self.available_numbers.any():
            # No numbers left and running total is not zero
            self.done = True
            # Current player wins (last to make a valid move)
            reward = 1  # Positive reward for winning
            return self._get_obs(), reward, self.done, False, self.info

        # Switch to the next player
        self.current_player *= -1
        reward = 0  # No reward for a valid move that doesn't end the game
        return self._get_obs(), reward, self.done, False, self.info

    def render(self):
        # Create a visual representation of the game state
        numbers_left = [i + 1 for i in range(9) if self.available_numbers[i] == 1]
        s = f"Number Pool: {numbers_left}\n"
        s += f"Running Total: {self.running_total}\n"
        s += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return s

    def valid_moves(self):
        # Return a list of valid actions based on available numbers
        moves = []
        for i in range(9):
            if self.available_numbers[i] == 1:
                # Both add and subtract operations are valid for available numbers
                moves.append(2 * i)  # Action to add number i+1
                moves.append(2 * i + 1)  # Action to subtract number i+1
        return moves
