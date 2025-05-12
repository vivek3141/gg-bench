import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to subtracting divisors from 2 to 50 (inclusive)
        self.action_space = spaces.Discrete(
            49
        )  # Actions from 0 to 48 correspond to divisors 2 to 50

        # Observation space contains the current player's score and the opponent's score
        self.observation_space = spaces.Box(low=0, high=50, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = [50, 50]  # Starting scores for both players
        self.current_player = 0  # Player 0 starts
        self.done = False
        observation = np.array(
            [
                self.player_scores[self.current_player],
                self.player_scores[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(
                    [
                        self.player_scores[self.current_player],
                        self.player_scores[1 - self.current_player],
                    ],
                    dtype=np.int32,
                ),
                0,
                True,
                False,
                {},
            )

        divisor = (
            action + 2
        )  # Map action to divisor (actions 0-48 correspond to divisors 2-50)
        current_score = self.player_scores[self.current_player]

        # Check if the action is valid
        if (
            current_score % divisor != 0
            or divisor == 1
            or divisor == current_score
            or divisor > current_score
        ):
            self.done = True
            return (
                np.array(
                    [
                        self.player_scores[self.current_player],
                        self.player_scores[1 - self.current_player],
                    ],
                    dtype=np.int32,
                ),
                -10,
                True,
                False,
                {},
            )  # Invalid move

        # Valid move: subtract the divisor from the current player's score
        self.player_scores[self.current_player] -= divisor
        current_score = self.player_scores[self.current_player]

        # Check if current player has won by reducing their score to exactly zero
        if current_score == 0:
            self.done = True
            return (
                np.array(
                    [current_score, self.player_scores[1 - self.current_player]],
                    dtype=np.int32,
                ),
                1,
                True,
                False,
                {},
            )

        # Check if the opponent has any valid moves
        opponent_score = self.player_scores[1 - self.current_player]
        opponent_valid_moves = self._get_valid_moves(opponent_score)

        if not opponent_valid_moves:
            # Opponent cannot make a valid move; current player wins
            self.done = True
            return (
                np.array([current_score, opponent_score], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player = 1 - self.current_player

        observation = np.array(
            [
                self.player_scores[self.current_player],
                self.player_scores[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return observation, 0, False, False, {}

    def render(self):
        return (
            f"Player {self.current_player + 1}'s Turn:\n"
            f"Your Score: {self.player_scores[self.current_player]}\n"
            f"Opponent's Score: {self.player_scores[1 - self.current_player]}\n"
        )

    def valid_moves(self):
        current_score = self.player_scores[self.current_player]
        valid_moves = self._get_valid_moves(current_score)
        # Map valid divisors back to action indices (divisor - 2)
        valid_actions = [divisor - 2 for divisor in valid_moves]
        return valid_actions

    def _get_valid_moves(self, score):
        # Return a list of proper divisors for the given score
        if score <= 1:
            return []
        divisors = [d for d in range(2, score) if score % d == 0 and d != score]
        return divisors
