import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_N=100):
        super(CustomEnv, self).__init__()

        self.starting_N = starting_N

        # Define action space
        # Action IDs:
        # 0: Subtract 1 from N
        # Actions 1 to starting_N - 2: Divide N by (action ID + 1)
        # Total actions: starting_N - 1 (IDs 0 to starting_N - 2)
        self.action_space = spaces.Discrete(self.starting_N - 1)

        # Observation space: Current number N, ranging from 1 to starting_N
        self.observation_space = spaces.Box(
            low=1, high=self.starting_N, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "starting_N" in options:
            self.starting_N = options["starting_N"]
            # Update action and observation spaces based on new starting_N
            self.action_space = spaces.Discrete(self.starting_N - 1)
            self.observation_space = spaces.Box(
                low=1, high=self.starting_N, shape=(1,), dtype=np.int32
            )

        self.N = self.starting_N
        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                -10,
                True,
                False,
                {"invalid_move": True},
            )
        else:
            # Perform the action
            if action == 0:
                # Subtract 1 from N
                self.N -= 1
            else:
                # Divide N by the selected proper divisor
                divisor = action + 1  # Mapping action ID to divisor
                self.N = self.N // divisor

            if self.N == 1:
                # Current player loses
                self.done = True
                reward = 1  # Opponent forced to reduce N to 1
            else:
                # Game continues
                reward = -10  # Valid move made
                self.current_player *= -1  # Switch player

            return np.array([self.N], dtype=np.int32), reward, self.done, False, {}

    def render(self):
        return f"Current N: {self.N}, Current Player: {self.current_player}"

    def valid_moves(self):
        if self.N <= 1:
            return []
        proper_divisors = [d for d in range(2, self.N) if self.N % d == 0]
        if len(proper_divisors) == 0:
            # N is prime; only valid action is to subtract 1
            return [0]
        else:
            # Valid actions are action IDs corresponding to proper divisors
            action_ids = [
                d - 1 for d in proper_divisors
            ]  # Mapping divisor to action ID
            return action_ids
