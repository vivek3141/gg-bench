import numpy as np
import gym
from gym import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Action space: Integers from 0 to 8 correspond to adding numbers 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # Index 0: Current total (0 to 50)
        # Indices 1-10: Flags for used last digits 0 to 9 (0 or 1)
        # Shape: (11,)
        obs_low = np.array([0] + [0] * 10)
        obs_high = np.array([50] + [1] * 10)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 0
        self.used_last_digits = [0] * 10  # Flags for last digits 0-9
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        info = {}
        if self.done:
            # The game has ended
            return self._get_observation(), 0, True, False, info

        # Convert action (0-8) to number to add (1-9)
        number_to_add = action + 1
        new_total = self.total + number_to_add

        # Check if the move is invalid due to exceeding 50
        if new_total > 50:
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        new_last_digit = new_total % 10
        # Check if the last digit has already been used
        if self.used_last_digits[new_last_digit] == 1:
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        # Valid move: update game state
        self.total = new_total
        self.used_last_digits[new_last_digit] = 1

        # Check for winning condition: reaching exactly 50
        if self.total == 50:
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, info

        # Check if the next player has any valid moves
        if not self._opponent_has_valid_moves():
            # Current player wins because opponent cannot move
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, info

        # Switch to the other player for the next turn
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        reward = 0  # No immediate reward
        return self._get_observation(), reward, False, False, info

    def render(self):
        # Return a string representation of the game state
        output = f"Current Total: {self.total}\n"
        output += f"Used Last Digits: {[i for i, used in enumerate(self.used_last_digits) if used == 1]}\n"
        output += f"Current Player: Player {self.current_player}\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices (0-8)
        valid_actions = []
        for action in range(9):
            number_to_add = action + 1
            new_total = self.total + number_to_add
            if new_total > 50:
                continue  # Cannot exceed 50
            new_last_digit = new_total % 10
            if self.used_last_digits[new_last_digit] == 0:
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Construct the observation array
        return np.array([self.total] + self.used_last_digits, dtype=np.int32)

    def _opponent_has_valid_moves(self):
        # Check if the opponent has any valid moves
        for action in range(9):
            number_to_add = action + 1
            potential_total = self.total + number_to_add
            if potential_total > 50:
                continue
            potential_last_digit = potential_total % 10
            if self.used_last_digits[potential_last_digit] == 0:
                return True  # Opponent has at least one valid move
        return False  # Opponent cannot make a valid move
