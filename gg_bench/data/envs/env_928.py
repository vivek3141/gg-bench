import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are integers from 0 to 8 corresponding to numbers 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation space consists of cumulative sum (0 to 23)
        # and opponent's last number (0 to 9)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([23, 9]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_sum = 0
        self.last_number = [0, 0]  # last_number[0]: Player 0, last_number[1]: Player 1
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self.get_observation(), {}  # Return observation and info

    def get_observation(self):
        # Observation includes the cumulative sum and opponent's last number
        opponent_last_number = self.last_number[1 - self.current_player]
        return np.array([self.cumulative_sum, opponent_last_number], dtype=np.int32)

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        opponent_last_number = self.last_number[1 - self.current_player]
        valid_actions = []
        for i in range(9):
            number = i + 1
            # Exclude the opponent's last number
            if number == opponent_last_number:
                continue
            # Exclude numbers that would exceed the cumulative sum limit
            if self.cumulative_sum + number > 23:
                continue
            valid_actions.append(i)
        return valid_actions

    def step(self, action):
        if self.done:
            # If the game is already over, no further moves are allowed
            return self.get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move: either the number was just played by the opponent
            # or the move exceeds the cumulative sum limit
            reward = -10
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        number = action + 1  # Map action index to the actual number (1-9)
        self.cumulative_sum += number
        self.last_number[self.current_player] = number

        if self.cumulative_sum == 23:
            # Current player wins by reaching the exact sum of 23
            reward = 1
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        # Switch players
        self.current_player = 1 - self.current_player

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Next player cannot make a valid move; current player wins
            self.current_player = 1 - self.current_player  # Switch back to the winner
            reward = 1
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        # Game continues
        reward = 0
        return self.get_observation(), reward, self.done, False, {}

    def render(self):
        opponent_last_number = self.last_number[1 - self.current_player]
        return (
            f"Current Sum: {self.cumulative_sum}, "
            f"Current Player: {self.current_player + 1}, "
            f"Opponent's Last Number: {opponent_last_number}"
        )
