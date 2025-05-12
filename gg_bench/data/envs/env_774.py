import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: actions 0 - 7 correspond to multipliers 2 - 9
        self.action_space = spaces.Discrete(8)

        # Observation space: [shared_number, current_player]
        # shared_number ranges from 1 to max_shared_number
        # current_player is 1 or 2
        self.max_shared_number = 1e12
        self.observation_space = spaces.Box(
            low=np.array([1, 1]),
            high=np.array([self.max_shared_number, 2]),
            dtype=np.float64,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.shared_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array(
            [self.shared_number, self.current_player], dtype=np.float64
        )
        return observation, {}

    def step(self, action):
        terminated = False
        truncated = False

        if not self.action_space.contains(action):
            # Invalid action
            reward = -100
            terminated = True
            observation = np.array(
                [self.shared_number, self.current_player], dtype=np.float64
            )
            return observation, reward, terminated, truncated, {}

        if self.done:
            # Game already over
            observation = np.array(
                [self.shared_number, self.current_player], dtype=np.float64
            )
            return observation, 0, True, False, {}

        # Map action (0-7) to multiplier (2-9)
        multiplier = action + 2

        # Apply the action
        self.shared_number *= multiplier

        # Check for win condition
        if self.shared_number % 10 == 0:
            # Current player wins
            reward = 1
            terminated = True
            self.done = True
        else:
            # Valid move, switch player
            reward = -10
            self.current_player = 2 if self.current_player == 1 else 1

        observation = np.array(
            [self.shared_number, self.current_player], dtype=np.float64
        )
        return observation, reward, terminated, truncated, {}

    def render(self):
        return f"Shared Number is {self.shared_number}\nCurrent Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        return list(range(8))
