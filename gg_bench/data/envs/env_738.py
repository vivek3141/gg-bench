import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum number for N and action space
        self.MAX_N = 100
        self.STARTING_N = 30

        # Define action and observation space
        # Actions are integers from 0 to MAX_N (0 represents pass)
        self.action_space = spaces.Discrete(self.MAX_N + 1)
        # Observation is the current N
        self.observation_space = spaces.Box(
            low=1, high=self.MAX_N, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.STARTING_N
        self.current_player = 1  # Can be 1 or -1
        self.done = False
        self.last_move_was_pass = False
        self.last_player_with_action = self.current_player
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.current_player *= -1  # Switch player
            return self._get_obs(), reward, self.done, False, {}

        if action == 0:
            # Pass action
            if self.last_move_was_pass:
                # Both players passed consecutively
                self.done = True
                if self.last_player_with_action == self.current_player:
                    reward = -1  # Current player loses
                else:
                    reward = 1  # Current player wins
                return self._get_obs(), reward, self.done, False, {}
            else:
                self.last_move_was_pass = True
                reward = 0
                self.current_player *= -1  # Switch player
                return self._get_obs(), reward, self.done, False, {}
        else:
            # Valid move
            self.N = int(self.N / action)
            self.last_move_was_pass = False
            self.last_player_with_action = self.current_player
            if self.N == 1:
                self.done = True
                reward = 1  # Current player wins
                return self._get_obs(), reward, self.done, False, {}
            else:
                reward = 0
                self.current_player *= -1  # Switch player
                return self._get_obs(), reward, self.done, False, {}

    def render(self):
        if self.done:
            game_status = "Game Over."
        else:
            game_status = f"Player {1 if self.current_player == 1 else 2}'s turn."
        render_str = (
            f"Current N: {self.N}\n"
            f"{game_status}\n"
            f"Proper divisors of {self.N}: {self._get_proper_divisors(self.N)}"
        )
        return render_str

    def valid_moves(self):
        if self.N == 1:
            return []
        divisors = self._get_proper_divisors(self.N)
        if not divisors:
            return [0]  # Only pass action is valid
        else:
            return divisors

    def _get_obs(self):
        return np.array([self.N], dtype=np.int32)

    def _get_proper_divisors(self, n):
        divisors = [i for i in range(2, n) if n % i == 0]
        return divisors
