import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10, K=3):
        super(CustomEnv, self).__init__()

        self.N = N  # Modulus Number
        self.K = K  # Maximum Add Number

        # Define action space: Integers from 0 to K-1
        self.action_space = spaces.Discrete(self.K)

        # Define observation space: Counter (0 to N-1) and Current Player (-1 or 1)
        self.observation_space = spaces.Box(
            low=np.array([0, -1]),
            high=np.array([self.N - 1, 1]),
            shape=(2,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        self.current_player = 1  # Player 1 starts the game
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_obs(), -10, True, False, {}

        if not self.action_space.contains(action):
            self.done = True
            # Invalid action
            return self._get_obs(), -10, True, False, {}

        # Map action index to actual number added (1 to K)
        a = action + 1
        self.counter = (self.counter + a) % self.N

        if self.counter == 0:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}
        else:
            # Switch to the other player
            self.current_player *= -1
            return self._get_obs(), 0, False, False, {}

    def render(self):
        return f"Current Counter: {self.counter}, Current Player: {self.current_player}"

    def valid_moves(self):
        return [i for i in range(self.action_space.n)]
