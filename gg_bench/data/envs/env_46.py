import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 8 possible multipliers: 2 to 9
        self.action_space = spaces.Discrete(8)

        # Observation space is the current number in the game, ranging from 1 to 1e10
        self.observation_space = spaces.Box(
            low=1.0, high=1e10, shape=(1,), dtype=np.float64
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1
        self.done = False
        return (
            np.array([self.current_number], dtype=np.float64),
            {},
        )  # observation, info

    def step(self, action):
        # Validate action
        if not self.action_space.contains(action):
            reward = -10
            self.done = True
            return (
                np.array([self.current_number], dtype=np.float64),
                reward,
                self.done,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

        # Map action to multiplier (2-9)
        multiplier = action + 2

        # Calculate new current number
        new_number = self.current_number * multiplier

        # Check for loss condition
        if new_number % 10 == 0:
            # Current player loses
            reward = -10
            self.done = True
            observation = np.array([new_number], dtype=np.float64)
            return observation, reward, self.done, False, {}
        else:
            # Game continues
            self.current_number = new_number
            reward = 0
            self.done = False
            # Switch to the next player
            self.current_player = 2 if self.current_player == 1 else 1
            observation = np.array([self.current_number], dtype=np.float64)
            return observation, reward, self.done, False, {}

    def render(self):
        # Return a string representation of the current state
        render_str = f"Current Number: {self.current_number}\n"
        render_str += f"Player {self.current_player}'s turn."
        return render_str

    def valid_moves(self):
        # All multipliers from 2 to 9 are valid at any time
        return list(range(8))  # Actions 0 to 7 correspond to multipliers 2 to 9
