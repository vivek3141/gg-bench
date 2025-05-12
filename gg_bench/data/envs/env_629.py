import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 2 to 20 (indices 0 to 18)
        self.action_space = spaces.Discrete(19)

        # Define observation space
        # Observation vector length: 20
        # Index 0: last number in chain (0 if chain is empty)
        # Indices 1-19: pool status (1 if number is available, 0 if used)
        self.observation_space = spaces.Box(low=0, high=20, shape=(20,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool and chain
        self.pool = [i for i in range(2, 21)]  # Numbers 2 to 20
        self.chain = []
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self.get_observation()
        info = {}
        return observation, info  # Return observation and info

    def get_observation(self):
        # Last number in the chain (0 if chain is empty)
        if self.chain:
            last_number = self.chain[-1]
        else:
            last_number = 0

        # Pool status: 1 if available, 0 if used
        pool_status = np.array(
            [1 if number in self.pool else 0 for number in range(2, 21)],
            dtype=np.int8,
        )

        observation = np.concatenate(([last_number], pool_status))
        return observation

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        # Check if current player has valid moves
        if not self.valid_moves():
            # Current player cannot make a move, loses
            self.done = True
            return self.get_observation(), -1, True, False, {}

        # Map action to number (action indices 0-18 correspond to numbers 2-20)
        number = action + 2

        # Check if number is in the pool
        if number not in self.pool:
            # Invalid move
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Check if the move is valid
        if not self.chain:
            valid = True  # Any number is valid if chain is empty
        else:
            last_number = self.chain[-1]
            if number % last_number == 0 or last_number % number == 0:
                valid = True
            else:
                valid = False

        if not valid:
            # Invalid move
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Apply action
        self.pool.remove(number)
        self.chain.append(number)

        # Check if opponent has valid moves
        if not self.has_valid_moves():
            # Opponent cannot move, current player wins
            self.done = True
            return self.get_observation(), 1, True, False, {}

        # Switch current player
        self.current_player = 1 if self.current_player == 2 else 2

        return self.get_observation(), 0, False, False, {}

    def has_valid_moves(self):
        if not self.pool:
            return False
        last_number = self.chain[-1]
        for num in self.pool:
            if num % last_number == 0 or last_number % num == 0:
                return True
        return False

    def valid_moves(self):
        if self.done:
            return []
        if not self.chain:
            # All numbers in pool are valid if chain is empty
            return [number - 2 for number in self.pool]
        else:
            last_number = self.chain[-1]
            valid_moves = []
            for number in self.pool:
                if number % last_number == 0 or last_number % number == 0:
                    valid_moves.append(number - 2)  # Convert number to action index
            return valid_moves

    def render(self):
        result = f"Current player: {self.current_player}\n"
        result += f"Chain: {self.chain}\n"
        result += "Available numbers in pool:\n"
        for number in self.pool:
            result += f"{number} "
        result += "\n"
        return result
