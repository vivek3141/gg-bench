import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_score=23):
        super(CustomEnv, self).__init__()

        self.target_score = target_score

        # Define action space: numbers 1 to 9, indices 0 to 8
        self.action_space = spaces.Discrete(9)

        # Define observation space:
        # First 9 elements: available numbers (0 or 1)
        # Next 2 elements: current player's score, opponent's score
        low = np.array([0] * 9 + [0, 0], dtype=np.int32)
        high = np.array(
            [1] * 9 + [self.target_score, self.target_score], dtype=np.int32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = [1] * 9  # Indices correspond to numbers 1-9
        self.current_player = 1  # Player 1 starts
        self.player_scores = {1: 0, -1: 0}
        self.done = False
        # Prepare initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if current player has any valid moves
        valid_moves = self._valid_moves()
        if not valid_moves:
            # Current player cannot make a valid move, they lose
            self.done = True
            return self._get_observation(), -1, True, False, {}

        # Check if action is valid
        if action not in valid_moves:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        number = action + 1  # Map action index to number

        # Apply the action
        self.player_scores[self.current_player] += number
        self.available_numbers[action] = 0  # Remove number from available numbers

        # Check for winning condition
        if self.player_scores[self.current_player] == self.target_score:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch players
        self.current_player *= -1

        # After switching, check if the new current player has any valid moves
        if not self._valid_moves():
            # Opponent cannot make a valid move, current player wins
            self.done = True
            # Switch back to original player to provide correct observation
            self.current_player *= -1
            return self._get_observation(), 1, True, False, {}
        else:
            # Continue game
            observation = self._get_observation()
            return observation, 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        available_numbers_str = " ".join(
            [str(i + 1) for i in range(9) if self.available_numbers[i]]
        )
        s = f"Target Score: {self.target_score}\n"
        s += f"Available Numbers: {available_numbers_str}\n"
        s += f"Player {1} Score: {self.player_scores[1]}\n"
        s += f"Player {-1} Score: {self.player_scores[-1]}\n"
        s += f"Current Player: Player {self.current_player}\n"
        return s

    def _get_observation(self):
        obs = np.array(
            self.available_numbers
            + [
                self.player_scores[self.current_player],
                self.player_scores[-self.current_player],
            ],
            dtype=np.int32,
        )
        return obs

    def _valid_moves(self):
        # Returns a list of valid actions for the current player
        valid_moves = []
        for action in range(9):
            if self.available_numbers[action] == 1:
                number = action + 1
                if (
                    self.player_scores[self.current_player] + number
                    <= self.target_score
                ):
                    valid_moves.append(action)
        return valid_moves

    def valid_moves(self):
        return self._valid_moves()
