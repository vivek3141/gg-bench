import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Bids from 1 to 10 (represented as indices 0-9)
        self.action_space = spaces.Discrete(10)

        # Observation space includes:
        # - Player's unused numbers (10 integers: 1 if unused, 0 if used)
        # - Opponent's used numbers (10 integers: 1 if used, 0 if unused)
        # - Player's score (integer)
        # - Opponent's score (integer)
        self.observation_space = spaces.Box(low=0, high=60, shape=(22,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the player's and opponent's numbers
        self.player_numbers = [1] * 10  # Player's numbers from 1 to 10 (unused)
        self.opponent_numbers = [1] * 10  # Opponent's numbers from 1 to 10 (unused)
        self.player_score = 0
        self.opponent_score = 0
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        # Convert action index (0-9) to bid number (1-10)
        bid = action + 1

        if self.done:
            return self._get_obs(), 0, True, False, {}

        if self.player_numbers[action] == 0:
            # Invalid move: Number already used
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Mark the bid as used
        self.player_numbers[action] = 0

        # Opponent's turn: Randomly select a bid from available numbers
        opponent_available_numbers = [
            idx for idx, val in enumerate(self.opponent_numbers) if val == 1
        ]

        if not opponent_available_numbers:
            # Opponent has no numbers left
            self.done = True
            if self.player_score > self.opponent_score:
                return self._get_obs(), 1, True, False, {}
            elif self.player_score < self.opponent_score:
                return self._get_obs(), -1, True, False, {}
            else:
                return self._get_obs(), 0, True, False, {}

        opponent_action = self.np_random.choice(opponent_available_numbers)
        opponent_bid = opponent_action + 1
        self.opponent_numbers[opponent_action] = 0  # Mark as used

        # Determine winner of the round
        if bid > opponent_bid:
            # Player wins the round
            self.player_score += bid + opponent_bid
        elif opponent_bid > bid:
            # Opponent wins the round
            self.opponent_score += bid + opponent_bid
        # Tie: No points awarded, bids are discarded

        # Check for game end
        if self.player_score >= 50 or self.opponent_score >= 50:
            self.done = True
            if self.player_score > self.opponent_score:
                return self._get_obs(), 1, True, False, {}
            elif self.player_score < self.player_score:
                return self._get_obs(), -1, True, False, {}
            else:
                return self._get_obs(), 0, True, False, {}
        else:
            return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        # Observation includes:
        # - Player's unused numbers (10 elements)
        # - Opponent's used numbers (10 elements)
        # - Player's score
        # - Opponent's score
        obs = np.array(
            self.player_numbers
            + [1 - num for num in self.opponent_numbers]
            + [self.player_score, self.opponent_score],
            dtype=np.int32,
        )
        return obs

    def render(self):
        # Return a string representation of the game state
        player_unused = [i + 1 for i, val in enumerate(self.player_numbers) if val == 1]
        opponent_used = [
            i + 1 for i, val in enumerate(self.opponent_numbers) if val == 0
        ]
        state = f"Player's unused numbers: {player_unused}\n"
        state += f"Opponent's used numbers: {opponent_used}\n"
        state += (
            f"Scores - Player: {self.player_score}, Opponent: {self.opponent_score}\n"
        )
        return state

    def valid_moves(self):
        # Return a list of valid moves (indices of unused numbers)
        return [idx for idx, val in enumerate(self.player_numbers) if val == 1]
