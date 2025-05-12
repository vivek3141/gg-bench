import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: integers 0-9 corresponding to divisors 1-10
        self.action_space = spaces.Discrete(10)

        # Observation space: player1_score, player2_score, current_player
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1]),
            high=np.array([50, 50, 1]),
            shape=(3,),
            dtype=np.int32,
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Both players start with score 50
        self.scores = {1: 50, -1: 50}
        self.current_player = 1  # 1 or -1
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action index to divisor
        divisor = action + 1  # Actions are 0-9, divisors are 1-10

        current_score = self.scores[self.current_player]

        # Check if action is valid
        if divisor < 1 or divisor > 10 or current_score % divisor != 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Apply valid move
        self.scores[self.current_player] -= divisor
        current_score = self.scores[self.current_player]

        # Check if current player won
        if current_score == 0:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        # Check if new current player has any valid moves
        if not self.valid_moves():
            # The previous player wins
            self.done = True
            # Switch back to previous player to return observation from their perspective
            self.current_player *= -1
            return self._get_observation(), 1, True, False, {}

        # Continue game
        return self._get_observation(), 0, False, False, {}

    def render(self):
        output = f"Player {self.current_player}'s turn.\n"
        output += f"Player 1's score: {self.scores[1]}\n"
        output += f"Player -1's score: {self.scores[-1]}\n"
        return output

    def valid_moves(self):
        current_score = self.scores[self.current_player]
        # Return valid action indices (0-9 corresponding to divisors 1-10)
        return [i for i in range(10) if current_score % (i + 1) == 0]

    def _get_observation(self):
        # Observation: [player1_score, player2_score, current_player]
        return np.array(
            [self.scores[1], self.scores[-1], self.current_player], dtype=np.int32
        )
