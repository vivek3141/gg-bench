import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space: 0-9 for Add 1-10, 10 for Multiply
        self.action_space = spaces.Discrete(11)
        # Observation space: Cumulative total from 0 to 50
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([50]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_total = 0
        self.current_player = 1  # Player 1 starts
        self.terminated = False
        return (
            np.array([self.cumulative_total], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action: current player loses
            reward = -10
            self.terminated = True
            return (
                np.array([self.cumulative_total], dtype=np.int32),
                reward,
                self.terminated,
                False,
                {},  # Observation, reward, terminated, truncated, info
            )

        # Apply the action
        if 0 <= action <= 9:
            # Add 1 to 10
            add_value = action + 1
            self.cumulative_total += add_value
        elif action == 10:
            # Multiply by 2
            self.cumulative_total *= 2

        # Check for win condition
        if self.cumulative_total == 50:
            # Current player wins
            reward = 1
            self.terminated = True
            return (
                np.array([self.cumulative_total], dtype=np.int32),
                reward,
                self.terminated,
                False,
                {},
            )

        # Switch to next player
        self.current_player = 3 - self.current_player  # Switches between 1 and 2

        # Check if next player has any valid moves
        if len(self.valid_moves()) == 0:
            # Current player wins because the opponent cannot move
            reward = 1
            self.terminated = True
            return (
                np.array([self.cumulative_total], dtype=np.int32),
                reward,
                self.terminated,
                False,
                {},
            )

        # Game continues
        reward = -10
        return (
            np.array([self.cumulative_total], dtype=np.int32),
            reward,
            self.terminated,
            False,
            {},
        )

    def render(self):
        return f"Cumulative Total: {self.cumulative_total}, Player {self.current_player}'s turn"

    def valid_moves(self):
        valid_actions = []
        # Actions for adding numbers from 1 to 10
        for action in range(10):
            add_value = action + 1
            if self.cumulative_total + add_value <= 50:
                valid_actions.append(action)
        # Action for multiplying by 2
        if self.cumulative_total * 2 <= 50:
            valid_actions.append(10)
        return valid_actions
