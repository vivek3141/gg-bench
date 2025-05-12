import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            5
        )  # Actions correspond to fractions indices 0-4

        # Observation space: 7 elements
        # Elements 0-4: fraction availability (1 if available, 0 if not)
        # Element 5: current player's cumulative sum (between 0 and 1)
        # Element 6: opponent's cumulative sum (between 0 and 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Fraction values corresponding to actions 0-4
        self.fraction_values = np.array(
            [0.5, 1 / 3, 0.25, 0.2, 1 / 6], dtype=np.float32
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Fractions availability: 1 if available, 0 if not
        self.fractions_available = np.ones(5, dtype=np.float32)
        # Players' cumulative sums: index 0 for player 0, index 1 for player 1
        self.player_sums = np.zeros(2, dtype=np.float32)
        # Current player: 0 or 1
        self.current_player = 0
        # Consecutive passes counter
        self.consecutive_passes = 0
        # Last player who made a valid move
        self.last_valid_player = None
        # Game done flag
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if current player has valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            self.consecutive_passes += 1
            if self.consecutive_passes >= 2:
                # Game ends, determine winner
                self.done = True
                winner = self._determine_winner()
                reward = 1 if winner == self.current_player else 0
                return self._get_obs(), reward, True, False, {}
            else:
                # Pass turn to the next player
                self.current_player = 1 - self.current_player
                return self._get_obs(), 0, False, False, {}

        else:
            # Reset consecutive passes since player has valid move
            self.consecutive_passes = 0
            # Check if action is valid
            if action not in valid_moves:
                self.done = True
                return self._get_obs(), -10, True, False, {}
            # Process the action
            fraction_value = self.fraction_values[action]
            self.fractions_available[action] = 0
            self.player_sums[self.current_player] += fraction_value
            self.last_valid_player = self.current_player
            if np.isclose(self.player_sums[self.current_player], 1.0):
                # Current player wins by reaching sum of exactly 1
                self.done = True
                return self._get_obs(), 1, True, False, {}
            else:
                # Check if player's sum exceeds 1 (should not happen)
                if self.player_sums[self.current_player] > 1.0:
                    self.done = True
                    return self._get_obs(), -10, True, False, {}
                # Pass turn to the next player
                self.current_player = 1 - self.current_player
                return self._get_obs(), 0, False, False, {}

    def render(self):
        fraction_symbols = ["1/2", "1/3", "1/4", "1/5", "1/6"]
        available_fractions = [
            fraction_symbols[i] for i in range(5) if self.fractions_available[i]
        ]
        state_representation = f"Player {self.current_player + 1}'s turn.\n"
        state_representation += f"Available fractions: {', '.join(available_fractions) if available_fractions else 'None'}\n"
        state_representation += f"Player 1 cumulative sum: {self.player_sums[0]:.4f}\n"
        state_representation += f"Player 2 cumulative sum: {self.player_sums[1]:.4f}\n"
        return state_representation

    def valid_moves(self):
        moves = []
        for i in range(5):
            if self.fractions_available[i]:
                fraction_value = self.fraction_values[i]
                if self.player_sums[self.current_player] + fraction_value <= 1.0 + 1e-8:
                    moves.append(i)
        return moves

    def _get_obs(self):
        obs = np.zeros(7, dtype=np.float32)
        obs[0:5] = self.fractions_available
        obs[5] = self.player_sums[self.current_player]
        obs[6] = self.player_sums[1 - self.current_player]
        return obs

    def _determine_winner(self):
        p1_sum = self.player_sums[0] if self.player_sums[0] <= 1.0 else -1.0
        p2_sum = self.player_sums[1] if self.player_sums[1] <= 1.0 else -1.0
        if p1_sum > p2_sum:
            return 0
        elif p2_sum > p1_sum:
            return 1
        else:
            # Sums are equal or both exceed 1, winner is last valid player
            return self.last_valid_player
