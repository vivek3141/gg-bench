import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Selecting numbers from 1 to 10 (indices 0 to 9)
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # - First 10 elements: Claimed numbers (0: unclaimed, 1: claimed)
        # - Next element: Last number selected (0 if none)
        # - Next element: Trend indicator (-1: downward, 0: no trend, 1: upward)
        self.observation_space = spaces.Box(
            low=np.array([0] * 10 + [0, -1]),
            high=np.array([1] * 10 + [10, 1]),
            dtype=np.int8,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.claimed_numbers = np.zeros(10, dtype=np.int8)  # Numbers 1 to 10
        self.last_number_selected = 0  # No number selected yet
        self.previous_number_selected = 0  # Second last number
        self.trend = 0  # 0: no trend, 1: upward, -1: downward
        self.done = False
        self.current_player = np.random.choice([1, -1])
        self.move_count = 0
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        number = action + 1  # Map action index to number (1-10)

        # Check if number has been claimed
        if self.claimed_numbers[action] == 1:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Game logic
        if self.move_count == 0:
            # First move: Any number is valid
            valid_move = True
        elif self.move_count == 1:
            # Second move: Must be different from first number
            if number != self.last_number_selected:
                valid_move = True
            else:
                # Invalid move
                self.done = True
                return self._get_obs(), -10, True, False, {}
            # Establish trend
            if number > self.last_number_selected:
                self.trend = 1  # Upward trend
            elif number < self.last_number_selected:
                self.trend = -1  # Downward trend
        else:
            # Subsequent moves: Must alternate trend
            required_trend = -self.trend
            if required_trend == 1 and number > self.last_number_selected:
                valid_move = True
            elif required_trend == -1 and number < self.last_number_selected:
                valid_move = True
            else:
                # Invalid move
                self.done = True
                return self._get_obs(), -10, True, False, {}

            # Alternate trend
            self.trend = -self.trend

        # Mark the number as claimed
        self.claimed_numbers[action] = 1

        # Update last numbers
        self.previous_number_selected = self.last_number_selected
        self.last_number_selected = number

        self.move_count += 1

        # Check if opponent can make a valid move
        opponent_can_move = False
        required_trend = -self.trend  # Opponent must alternate trend

        for i in range(10):
            if self.claimed_numbers[i] == 0:
                num = i + 1
                if required_trend == 1 and num > self.last_number_selected:
                    opponent_can_move = True
                    break
                elif required_trend == -1 and num < self.last_number_selected:
                    opponent_can_move = True
                    break

        if not opponent_can_move:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch player
        self.current_player *= -1

        return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.claimed_numbers,
                np.array([self.last_number_selected], dtype=np.int8),
                np.array([self.trend], dtype=np.int8),
            ]
        )
        return obs

    def render(self):
        claimed_numbers_str = ",".join(
            [str(i + 1) for i in range(10) if self.claimed_numbers[i] == 1]
        )
        available_numbers_str = ",".join(
            [str(i + 1) for i in range(10) if self.claimed_numbers[i] == 0]
        )
        trend_str = (
            "Upward"
            if self.trend == 1
            else "Downward" if self.trend == -1 else "No trend established"
        )
        last_num_str = (
            str(self.last_number_selected) if self.last_number_selected else "None"
        )

        render_str = (
            f"Claimed Numbers: {claimed_numbers_str}\n"
            f"Available Numbers: {available_numbers_str}\n"
            f"Trend: {trend_str}\n"
            f"Last Number Selected: {last_num_str}\n"
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )

        return render_str

    def valid_moves(self):
        valid_moves = []
        if self.done:
            return valid_moves

        if self.move_count == 0:
            # First move: Any unclaimed number
            valid_moves = [i for i in range(10) if self.claimed_numbers[i] == 0]
        elif self.move_count == 1:
            # Second move: Any unclaimed number different from the first
            valid_moves = [
                i
                for i in range(10)
                if self.claimed_numbers[i] == 0 and (i + 1) != self.last_number_selected
            ]
        else:
            required_trend = -self.trend
            last_number = self.last_number_selected
            if required_trend == 1:
                # Must select a higher number
                valid_moves = [
                    i
                    for i in range(10)
                    if self.claimed_numbers[i] == 0 and (i + 1) > last_number
                ]
            elif required_trend == -1:
                # Must select a lower number
                valid_moves = [
                    i
                    for i in range(10)
                    if self.claimed_numbers[i] == 0 and (i + 1) < last_number
                ]

        return valid_moves
