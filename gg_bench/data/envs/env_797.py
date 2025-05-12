import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            50
        )  # Numbers 1 to 50 represented as actions 0 to 49
        self.observation_space = spaces.Box(
            low=0, high=50, shape=(51,), dtype=np.int32
        )  # First 50 for unclaimed numbers, last for last number selected

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.unclaimed_numbers = np.ones(50, dtype=np.int32)  # Numbers from 1 to 50
        self.last_number = 0  # No number selected yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.concatenate(
            (self.unclaimed_numbers, np.array([self.last_number]))
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        action_number = action + 1  # Convert action index to actual number
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid move

        # Valid move, update game state
        self.unclaimed_numbers[action] = 0  # Mark number as claimed
        self.last_number = action_number  # Update last number selected

        # Check if the next player has any valid moves
        next_valid_moves = self._get_valid_moves_for_number(action_number)
        if not next_valid_moves:
            # The opponent cannot make a valid move, current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Switches between 1 and 2
        return self._get_observation(), 0, False, False, {}  # Continue game

    def render(self):
        available_numbers = [
            str(i + 1) for i in range(50) if self.unclaimed_numbers[i] == 1
        ]
        available_numbers_str = " ".join(available_numbers)
        game_state = f"Available Numbers: {available_numbers_str}\n"
        game_state += f"Last Number Selected: {self.last_number}\n"
        game_state += f"Current Player: Player {self.current_player}\n"
        return game_state

    def valid_moves(self):
        if self.last_number == 0:
            # First turn, all unclaimed numbers are valid
            return [i for i in range(50) if self.unclaimed_numbers[i] == 1]
        else:
            # Subsequent turns, numbers that are factors or multiples of the last number
            return self._get_valid_moves_for_number(self.last_number)

    def _get_observation(self):
        observation = np.concatenate(
            (self.unclaimed_numbers, np.array([self.last_number]))
        )
        return observation

    def _get_valid_moves_for_number(self, number):
        valid_moves = []
        for i in range(50):
            if self.unclaimed_numbers[i] == 1:
                candidate_number = i + 1
                if candidate_number % number == 0 or number % candidate_number == 0:
                    valid_moves.append(i)
        return valid_moves
