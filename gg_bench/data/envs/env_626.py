import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 50 possible actions (numbers 1 to 50)
        self.action_space = spaces.Discrete(50)

        # Observation space: [current_player_score, opponent_player_score, numbers_available (50 elements)]
        low_obs = np.array([0.0, 0.0] + [0.0] * 50, dtype=np.float32)
        high_obs = np.array([100.0, 100.0] + [1.0] * 50, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers_available = np.ones(
            50, dtype=np.float32
        )  # Numbers 1 to 50 are available
        self.player_scores = [0.0, 0.0]  # Scores for player 1 and player 2
        self.current_player = np.random.choice(
            [0, 1]
        )  # Randomly choose starting player
        self.done = False
        self.extra_turn = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}  # The game is over
        if action < 0 or action >= 50 or self.numbers_available[action] == 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        number_selected = action + 1  # Numbers are from 1 to 50
        self.numbers_available[action] = 0  # Remove number from pool
        self.player_scores[self.current_player] += number_selected

        # Check for score over 100
        if self.player_scores[self.current_player] > 100.0:
            self.player_scores[self.current_player] = 50.0

        # Check for win
        if self.player_scores[self.current_player] == 100.0:
            self.done = True
            return self._get_obs(), 1.0, True, False, {}

        # Check if number is prime
        if is_prime(number_selected):
            # Player gets an extra turn
            self.extra_turn = True
        else:
            self.extra_turn = False
            # Switch player
            self.current_player = 1 - self.current_player

        # Check if numbers are exhausted
        if np.sum(self.numbers_available) == 0:
            self.done = True
            # Check which player is closer to 100
            diff_current = abs(100.0 - self.player_scores[self.current_player])
            diff_opponent = abs(100.0 - self.player_scores[1 - self.current_player])
            if diff_current < diff_opponent:
                # Current player wins
                return self._get_obs(), 1.0, True, False, {}
            elif diff_current > diff_opponent:
                # Opponent wins
                return self._get_obs(), 0.0, True, False, {}
            else:
                # Tie
                return self._get_obs(), 0.0, True, False, {}

        return self._get_obs(), 0.0, False, False, {}

    def render(self):
        lines = []
        lines.append(f"Player {self.current_player + 1}'s turn")
        lines.append(f"Player 1 Score: {self.player_scores[0]}")
        lines.append(f"Player 2 Score: {self.player_scores[1]}")
        available_numbers = [
            str(i + 1) for i in range(50) if self.numbers_available[i] == 1
        ]
        lines.append("Available Numbers: " + ", ".join(available_numbers))
        return "\n".join(lines)

    def valid_moves(self):
        return [i for i in range(50) if self.numbers_available[i] == 1]

    def _get_obs(self):
        # Observation: [current_player_score, opponent_player_score, numbers_available (50 elements)]
        obs = np.array(
            [
                self.player_scores[self.current_player],
                self.player_scores[1 - self.current_player],
            ]
            + self.numbers_available.tolist(),
            dtype=np.float32,
        )
        return obs
