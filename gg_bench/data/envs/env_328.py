import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the numbers in the pool
        self.numbers = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=np.int32)

        # Define action and observation space
        self.action_space = spaces.Discrete(10)
        # Observation space includes the availability of numbers and the running totals
        # First 10 elements: availability of numbers (1 if available, 0 if taken)
        # Next 2 elements: current player's running total, opponent's running total
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (np.zeros(10, dtype=np.int32), np.array([-15, -15], dtype=np.int32))
            ),
            high=np.concatenate(
                (np.ones(10, dtype=np.int32), np.array([15, 15], dtype=np.int32))
            ),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.available = np.ones(10, dtype=np.int32)
        self.player_totals = [0, 0]  # Index 0: Player 1, Index 1: Player 2
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        self.winner = None
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action < 0 or action >= 10 or self.available[action] == 0:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid action
        # Update the running total of the current player
        selected_number = self.numbers[action]
        self.player_totals[self.current_player] += selected_number
        self.available[action] = 0  # Remove the number from the pool

        # Check for immediate win
        if self.player_totals[self.current_player] == 0:
            self.done = True
            self.winner = self.current_player
            return self._get_obs(), 1, True, False, {}

        # Check if all numbers are exhausted
        if np.all(self.available == 0):
            self.done = True
            # Determine winner based on who is closer to zero
            dist_current = abs(self.player_totals[self.current_player])
            dist_opponent = abs(self.player_totals[1 - self.current_player])
            if dist_current < dist_opponent:
                self.winner = self.current_player
                return self._get_obs(), 1, True, False, {}
            elif dist_current > dist_opponent:
                self.winner = 1 - self.current_player
                return self._get_obs(), -1, True, False, {}
            else:
                # Sudden death
                self.available = np.ones(10, dtype=np.int32)
                return self._get_obs(), 0, False, False, {}
        else:
            # Switch to the next player
            self.current_player = 1 - self.current_player
            return self._get_obs(), 0, False, False, {}

    def render(self):
        pool_state = "Available Numbers: "
        for i in range(len(self.numbers)):
            if self.available[i]:
                pool_state += f"{self.numbers[i]} "
        pool_state += "\n"
        pool_state += f"Player 1 Running Total: {self.player_totals[0]}\n"
        pool_state += f"Player 2 Running Total: {self.player_totals[1]}\n"
        pool_state += f"Current Player: {'Player 1' if self.current_player == 0 else 'Player 2'}\n"
        return pool_state

    def valid_moves(self):
        return [i for i in range(10) if self.available[i] == 1]

    def _get_obs(self):
        # Observation includes availability and running totals
        obs = np.concatenate(
            (
                self.available.copy(),
                np.array(
                    [
                        self.player_totals[self.current_player],
                        self.player_totals[1 - self.current_player],
                    ],
                    dtype=np.int32,
                ),
            )
        )
        return obs
