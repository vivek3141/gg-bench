import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is numbers 1 to 9 (action 0 maps to number 1)
        self.action_space = spaces.Discrete(9)
        # Observation space consists of:
        # - Last number in the chain (0 if chain is empty)
        # - Availability of numbers 1 to 9 (1 if available, 0 if used)
        self.observation_space = spaces.Box(
            low=np.array([0] + [0] * 9), high=np.array([9] + [1] * 9), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.chain = []
        self.available_numbers = np.ones(9, dtype=np.int32)  # Numbers 1-9 are available
        self.last_number = 0  # No last number at the start
        self.done = False
        self.current_player = 1  # Player 1 starts
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            # Game is over
            observation = self._get_observation()
            reward = 0
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        # Map action to chosen number
        number_chosen = action + 1  # Actions are 0-8, numbers are 1-9

        # Check if the number is available
        if self.available_numbers[number_chosen - 1] == 0:
            # Invalid move, number already used
            self.done = True
            observation = self._get_observation()
            reward = -10
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        # Check if the move is valid
        if len(self.chain) == 0:
            # First move, any number is valid
            valid_move = True
        else:
            # Subsequent moves, number must be factor or multiple of last number
            if (
                self.last_number % number_chosen == 0
                or number_chosen % self.last_number == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move according to game rules
            self.done = True
            observation = self._get_observation()
            reward = -10
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        # Valid move, update game state
        self.chain.append(number_chosen)
        self.available_numbers[number_chosen - 1] = 0  # Mark number as used
        self.last_number = number_chosen

        # Check if opponent has any valid moves
        opponent_valid_moves = self._get_valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot move, current player wins
            self.done = True
            observation = self._get_observation()
            reward = 1  # Current player wins
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        # Switch to next player (since agent plays both roles, we just proceed)
        self.current_player *= -1  # Switch player (-1 or 1)

        # Continue the game
        observation = self._get_observation()
        reward = 0  # No reward for regular move
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        if not self.chain:
            return "The number chain is empty."
        else:
            chain_str = ", ".join(map(str, self.chain))
            return f"Number chain: {chain_str}"

    def valid_moves(self):
        # Return valid moves for current player
        return self._get_valid_moves()

    def _get_observation(self):
        # Observation consists of last number and availability of numbers 1-9
        observation = np.zeros(10, dtype=np.int32)
        observation[0] = (
            self.last_number
        )  # Last number in the chain (0 if chain is empty)
        observation[1:] = self.available_numbers  # Availability of numbers 1-9
        return observation

    def _get_valid_moves(self):
        valid_moves = []
        for i in range(9):  # Numbers 1 to 9
            if self.available_numbers[i] == 1:
                number = i + 1
                if len(self.chain) == 0:
                    # Any number is valid on the first turn
                    valid_moves.append(i)
                else:
                    # Number must be factor or multiple of last number
                    if self.last_number % number == 0 or number % self.last_number == 0:
                        valid_moves.append(i)
        return valid_moves
