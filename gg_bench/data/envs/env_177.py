import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Digits 1 to 9
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(11,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Digits 1-9 are all available
        self.digits_status = np.zeros(
            9, dtype=np.float32
        )  # 0: available, 1: player A, -1: player B
        # Players' totals: index 0 for player A (1), index 1 for player B (-1)
        self.totals = np.array([0.0, 0.0], dtype=np.float32)
        # Current player: 1 (Player A) or -1 (Player B)
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation vector: digits_status[0..8], totals[0..1]
        obs = np.concatenate([self.digits_status, self.totals])
        return obs

    def valid_moves(self):
        """
        Returns a list of valid actions (indices) for the current player.
        """
        player_idx = 0 if self.current_player == 1 else 1
        current_total = self.totals[player_idx]
        valid_actions = []
        for i in range(9):
            if self.digits_status[i] == 0:
                digit = i + 1
                if current_total + digit <= 10:
                    valid_actions.append(i)
        return valid_actions

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        # Valid action
        player_idx = 0 if self.current_player == 1 else 1
        opponent_idx = 1 - player_idx
        digit = action + 1
        self.digits_status[action] = self.current_player
        self.totals[player_idx] += digit
        current_total = self.totals[player_idx]

        # Check for win condition
        if current_total == 10:
            self.done = True
            return self._get_obs(), 1.0, True, False, {}
        elif current_total > 10:
            self.done = True
            return self._get_obs(), -1.0, True, False, {}

        # Switch players
        self.current_player *= -1
        return self._get_obs(), 0.0, False, False, {}

    def render(self):
        player_symbols = {1: "A", -1: "B", 0: " "}
        digits_info = ""
        for i in range(9):
            status = self.digits_status[i]
            symbol = player_symbols.get(status, " ")
            digits_info += f"{i+1}:{symbol}  "
            if (i + 1) % 3 == 0:
                digits_info += "\n"
        info = (
            f"Player A's Total: {self.totals[0]}\n"
            f"Player B's Total: {self.totals[1]}\n"
            f"Current Player: {'A' if self.current_player == 1 else 'B'}\n"
            f"Available Moves: {self.valid_moves_display()}\n"
        )
        return digits_info + info

    def valid_moves_display(self):
        valid_actions = self.valid_moves()
        moves = [str(i + 1) for i in valid_actions]
        return " ".join(moves)
