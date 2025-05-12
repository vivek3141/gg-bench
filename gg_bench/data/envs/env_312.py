import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            9
        )  # Actions: choose numbers 1-9 (indices 0-8)
        self.observation_space = spaces.Box(low=0, high=99, shape=(11,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(9, dtype=np.int32)  # numbers 1-9 are available
        self.current_totals = [0, 0]  # Player 0 and Player 1 cumulative totals
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.terminated = False
        self.truncated = False
        return self._get_obs(), {}

    def step(self, action):
        if self.terminated:
            return self._get_obs(), -10, True, False, {}
        if action < 0 or action >= 9 or self.available_numbers[action] == 0:
            self.terminated = True
            return self._get_obs(), -10, True, False, {}
        # Valid action
        number_selected = action + 1  # numbers 1-9
        self.current_totals[self.current_player] += number_selected
        self.available_numbers[action] = 0  # Mark number as taken

        # Check for immediate win
        if self.current_totals[self.current_player] % 10 == 0:
            self.terminated = True
            return self._get_obs(), 1, True, False, {}

        # Check if all numbers have been selected:
        if np.sum(self.available_numbers) == 0:
            # Tiebreaker
            last_digit_current = self.current_totals[self.current_player] % 10
            last_digit_opponent = self.current_totals[1 - self.current_player] % 10
            distance_current = min(last_digit_current, 10 - last_digit_current)
            distance_opponent = min(last_digit_opponent, 10 - last_digit_opponent)

            if distance_current < distance_opponent:
                # Current player wins
                self.terminated = True
                return self._get_obs(), 1, True, False, {}
            elif distance_current > distance_opponent:
                # Opponent wins
                self.terminated = True
                return self._get_obs(), -1, True, False, {}
            else:
                # Equal proximity, check cumulative totals
                if (
                    self.current_totals[self.current_player]
                    < self.current_totals[1 - self.current_player]
                ):
                    # Current player wins
                    self.terminated = True
                    return self._get_obs(), 1, True, False, {}
                elif (
                    self.current_totals[self.current_player]
                    > self.current_totals[1 - self.current_player]
                ):
                    # Opponent wins
                    self.terminated = True
                    return self._get_obs(), -1, True, False, {}
                else:
                    # Draw (should not happen according to the game rules)
                    self.terminated = True
                    return self._get_obs(), 0, True, False, {}
        else:
            # Switch to other player
            self.current_player = 1 - self.current_player
            return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        # Observation: Available numbers, current player's total, opponent's total
        obs = np.zeros(11, dtype=np.int32)
        obs[0:9] = self.available_numbers
        obs[9] = self.current_totals[self.current_player]
        obs[10] = self.current_totals[1 - self.current_player]
        return obs

    def render(self):
        available_numbers = [i + 1 for i in range(9) if self.available_numbers[i] == 1]
        s = f"Available Numbers: {available_numbers}\n"
        s += f"Player {self.current_player + 1} Total: {self.current_totals[self.current_player]}\n"
        s += f"Player {2 - self.current_player} Total: {self.current_totals[1 - self.current_player]}\n"
        s += f"Player {self.current_player +1}'s turn\n"
        return s

    def valid_moves(self):
        return [i for i in range(9) if self.available_numbers[i] == 1]
