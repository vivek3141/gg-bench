import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_N=20, max_N=100):
        super(CustomEnv, self).__init__()

        self.initial_N = initial_N
        self.max_N = max_N

        # Define action and observation space
        # The actions are integers from 1 to max_N inclusive; indices from 0 to max_N-1
        self.action_space = spaces.Discrete(self.max_N)

        # The observation is the current N (integer between 1 and max_N)
        # The observation space is a Box with shape (1,)
        self.observation_space = spaces.Box(
            low=1, high=self.max_N, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.game_over = False
        # Return observation and info
        return np.array([self.N], dtype=np.int32), {}

    def get_proper_divisors(self, N):
        return [i for i in range(1, N) if N % i == 0]

    def valid_moves(self):
        # Return action indices corresponding to proper divisors of N
        divisors = self.get_proper_divisors(self.N)
        # Map divisors to action indices (action_value = action + 1)
        return [d - 1 for d in divisors]  # Adjust for 0-based indexing

    def step(self, action):
        if self.game_over:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        action_value = (
            action + 1
        )  # Map action index to actual action value (1 to max_N)

        # Check if action_value is a proper divisor of N
        if action_value < self.N and self.N % action_value == 0:
            # Valid move
            self.N -= action_value

            # Check if the next player has any valid moves
            if self.N == 1 or len(self.get_proper_divisors(self.N)) == 0:
                # Next player cannot move, current player wins
                reward = 1  # Current player wins
                self.game_over = True
                return np.array([self.N], dtype=np.int32), reward, True, False, {}
            else:
                # Switch to next player
                self.current_player *= -1
                return np.array([self.N], dtype=np.int32), 0, False, False, {}
        else:
            # Invalid move
            reward = -10  # Penalty for invalid move
            self.game_over = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

    def render(self):
        divisors = self.get_proper_divisors(self.N)
        state = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        state += f"Current N: {self.N}\n"
        state += f"Proper Divisors of N: {divisors}\n"
        return state
