import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Numbers from 2 to 50 inclusive
        self.numbers = np.arange(2, 51)
        self.n_numbers = len(self.numbers)  # Total of 49 numbers

        # Define action space: Discrete space of available numbers from 2 to 50
        self.action_space = spaces.Discrete(self.n_numbers)

        # Define observation space: Binary vector of number pool and current player
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.n_numbers + 1,), dtype=np.int8
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool: 1 means available, 0 means removed
        self.number_pool = np.ones(self.n_numbers, dtype=np.int8)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        observation = np.append(self.number_pool.copy(), self.current_player)
        return observation, {}

    def step(self, action):
        if self.done:
            observation = np.append(self.number_pool.copy(), self.current_player)
            return observation, 0, True, False, {}

        # Validate action
        if not self.action_space.contains(action) or self.number_pool[action] == 0:
            # Invalid action
            self.done = True
            observation = np.append(self.number_pool.copy(), self.current_player)
            return observation, -10, True, False, {}

        # Valid action: process the selected number
        selected_number = self.numbers[action]
        self.number_pool[action] = 0  # Remove selected number

        # Remove factors and multiples of the selected number from the pool
        for idx, num in enumerate(self.numbers):
            if self.number_pool[idx] == 1 and num != selected_number:
                if num % selected_number == 0 or selected_number % num == 0:
                    self.number_pool[idx] = 0  # Remove number

        # Check for win condition: no numbers left means current player wins
        if not any(self.number_pool == 1):
            self.done = True
            observation = np.append(self.number_pool.copy(), self.current_player)
            return observation, 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        observation = np.append(self.number_pool.copy(), self.current_player)
        return observation, 0, False, False, {}

    def render(self):
        # Visual representation of the game state
        available_numbers = self.numbers[self.number_pool == 1]
        board_str = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        board_str += "Available Numbers: " + ", ".join(map(str, available_numbers))
        return board_str

    def valid_moves(self):
        # List of valid moves as indices of action_space
        return [idx for idx, val in enumerate(self.number_pool) if val == 1]
