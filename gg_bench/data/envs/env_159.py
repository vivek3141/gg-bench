import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The numbers 1 to 100 are represented as actions 0 to 99
        self.action_space = spaces.Discrete(100)
        # Observation space consists of 102 elements:
        # - 100 elements for used numbers (0: unused, 1: used)
        # - 1 element for the previous number selected (scaled between 0 and 1)
        # - 1 element for the current player (0 or 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(102,), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.used_numbers = np.zeros(100, dtype=np.float32)  # Numbers from 1 to 100
        self.previous_number = None  # Previous number selected by opponent
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        number = action + 1  # Map action to number between 1 and 100

        # Check if action is valid
        if self.used_numbers[action] != 0 or not self._is_valid_move(number):
            # Invalid move
            reward = -10.0
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}
        else:
            # Valid move
            self.used_numbers[action] = 1.0  # Mark number as used
            self.previous_number = number

            # Switch to opponent
            self.current_player = 3 - self.current_player  # Switch player

            # Check if opponent has any valid moves
            if len(self.valid_moves()) == 0:
                # Opponent cannot move, current player wins
                reward = 1.0
                self.done = True
                # Switch back to current player for consistency
                self.current_player = 3 - self.current_player
                observation = self._get_observation()
                return observation, reward, True, False, {}
            else:
                # Continue game
                # Switch back to current player for consistency
                self.current_player = 3 - self.current_player
                observation = self._get_observation()
                return observation, 0.0, False, False, {}

    def _is_valid_move(self, number):
        # First move can be any unused number
        if self.previous_number is None:
            return True
        # Check if number is a multiple or factor of previous_number
        if number % self.previous_number == 0 or self.previous_number % number == 0:
            return True
        # Check if number shares a digit with previous_number
        if set(str(number)).intersection(str(self.previous_number)):
            return True
        return False

    def valid_moves(self):
        valid_moves = []
        for action in range(100):
            if self.used_numbers[action] == 0:
                number = action + 1
                if self._is_valid_move(number):
                    valid_moves.append(action)
        return valid_moves

    def render(self):
        used_numbers = [i + 1 for i in range(100) if self.used_numbers[i] == 1.0]
        game_state = f"Used numbers: {used_numbers}"
        prev_num = self.previous_number if self.previous_number is not None else "None"
        current_player = f"Player {self.current_player}"
        game_info = f"{current_player}'s turn. Previous number: {prev_num}"
        return game_state + "\n" + game_info

    def _get_observation(self):
        obs = np.zeros(102, dtype=np.float32)
        obs[0:100] = self.used_numbers
        if self.previous_number is not None:
            obs[100] = self.previous_number / 100.0  # Scale between 0 and 1
        else:
            obs[100] = 0.0
        obs[101] = float(self.current_player - 1)  # 0 for player 1, 1 for player 2
        return obs
