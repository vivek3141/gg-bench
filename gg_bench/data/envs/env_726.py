import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: actions 0-7 correspond to multipliers 2-9
        self.action_space = spaces.Discrete(8)

        # Define observation space
        # observation[0]: current number (from 1 to 10000)
        # observation[1]: current player (0 or 1)
        self.observation_space = spaces.Box(
            low=np.array([1, 0]), high=np.array([1e4, 1]), dtype=np.float32
        )

        self.current_number = None
        self.current_player = None
        self.done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 0  # Player 1
        self.done = False
        observation = np.array(
            [self.current_number, self.current_player], dtype=np.float32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            reward = 0
            terminated = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.float32
            )
            truncated = False
            return observation, reward, terminated, truncated, {}

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid action
            reward = -10
            self.done = True
            terminated = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.float32
            )
            truncated = False
            return observation, reward, terminated, truncated, {}

        # Map action to multiplier (0 -> 2, 1 -> 3, ..., 7 -> 9)
        multiplier = action + 2

        # Update current number
        self.current_number *= multiplier

        # Check for win condition
        if self.current_number >= 100:
            reward = 1
            self.done = True
            terminated = True
        else:
            reward = 0
            terminated = False

        # Switch current player
        self.current_player = 1 - self.current_player

        # Prepare observation
        observation = np.array(
            [self.current_number, self.current_player], dtype=np.float32
        )

        # No truncation in this environment
        truncated = False

        return observation, reward, terminated, truncated, {}

    def render(self):
        state_str = (
            f"Current number: {self.current_number}, "
            f"Current player: Player {self.current_player + 1}"
        )
        return state_str

    def valid_moves(self):
        return list(range(8))
