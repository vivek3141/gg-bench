import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete actions from 0 to 48, corresponding to numbers 2 to 50
        self.action_space = spaces.Discrete(49)

        # Observation space: 49 numbers status (0=available, 1=used), and current number index (-1 to 48)
        # First 49 elements represent the numbers' status, last element is the current number index
        low = np.array([0] * 49 + [-1], dtype=np.int32)
        high = np.array([1] * 49 + [48], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = set(range(2, 51))  # Numbers from 2 to 50
        self.used_numbers = set()
        self.current_number = -1  # No current number at the start
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Build initial observation
        obs = np.zeros(50, dtype=np.int32)
        obs[0:49] = 0  # All numbers are available
        obs[49] = -1  # No current number

        return obs, {}

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            obs = self._get_obs()
            return obs, 0, self.done, False, {}

        # Map action to the actual number (action 0 corresponds to number 2)
        selected_number = action + 2

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            obs = self._get_obs()
            return obs, reward, self.done, False, {}

        # Update game state
        self.number_pool.remove(selected_number)
        self.used_numbers.add(selected_number)
        self.current_number = selected_number

        # Switch current player
        self.current_player = 1 if self.current_player == 2 else 2

        # Check if the next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Opponent cannot make a valid move, current player wins
            self.done = True
            reward = 1
        else:
            # Game continues
            reward = 0

        # Build observation
        obs = self._get_obs()

        return obs, reward, self.done, False, {}

    def render(self):
        # Return a string representation of the current game state
        game_state = "Current Game State:\n"
        game_state += f"Current Player: Player {self.current_player}\n"
        game_state += f"Opponent's Last Number: {self.current_number if self.current_number != -1 else 'None'}\n"
        game_state += "Available Numbers: "
        available_numbers = sorted(self.number_pool)
        game_state += ", ".join(map(str, available_numbers)) + "\n"
        game_state += "Used Numbers: "
        used_numbers = sorted(self.used_numbers)
        game_state += ", ".join(map(str, used_numbers)) + "\n"
        return game_state

    def valid_moves(self):
        valid_actions = []
        if self.current_number == -1:
            # First move, all numbers are valid
            for number in self.number_pool:
                action = number - 2
                valid_actions.append(action)
        else:
            # Valid moves are factors or multiples of current_number
            for number in self.number_pool:
                if (
                    number % self.current_number == 0
                    or self.current_number % number == 0
                ):
                    action = number - 2
                    valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Build the observation vector
        obs = np.zeros(50, dtype=np.int32)
        for number in range(2, 51):
            index = number - 2
            if number in self.used_numbers:
                obs[index] = 1  # Number has been used
            else:
                obs[index] = 0  # Number is available
        # Set the current number index
        if self.current_number == -1:
            obs[49] = -1  # No current number
        else:
            obs[49] = self.current_number - 2  # Index from 0 to 48
        return obs
