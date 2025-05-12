import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space: 4 possible actions
        # 0: Increase by 1
        # 1: Increase by 2
        # 2: Decrease by 1
        # 3: Decrease by 2
        self.action_space = spaces.Discrete(4)
        # Define observation space: gravity counter value
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.int32
        )
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize gravity counter to 0
        self.gravity_counter = 0
        # Positive Player starts first
        self.current_player = 1  # 1 for Positive Player, -1 for Negative Player
        self.done = False
        return np.array([self.gravity_counter], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # Game is over
            return (
                np.array([self.gravity_counter], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Map action to movement
        if action == 0:
            move = 1
        elif action == 1:
            move = 2
        elif action == 2:
            move = -1
        elif action == 3:
            move = -2
        else:
            # Invalid action index
            self.done = True
            reward = -10
            return (
                np.array([self.gravity_counter], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Compute new gravity counter
        new_counter = self.gravity_counter + move

        # Check if new_counter is within bounds
        if new_counter < -10 or new_counter > 10:
            # Invalid move
            self.done = True
            reward = -10
            return (
                np.array([self.gravity_counter], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Update gravity counter
        self.gravity_counter = new_counter

        # Check for win condition
        if self.current_player == 1:
            # Positive Player wins if gravity_counter == +10
            if self.gravity_counter == 10:
                self.done = True
                reward = 1
                return (
                    np.array([self.gravity_counter], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
        else:
            # Negative Player wins if gravity_counter == -10
            if self.gravity_counter == -10:
                self.done = True
                reward = 1
                return (
                    np.array([self.gravity_counter], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )

        # Switch player
        self.current_player *= -1  # Toggle between 1 (Positive) and -1 (Negative)

        # Continue the game
        return (
            np.array([self.gravity_counter], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        # Return a string representation of the current gravity counter
        return f"Current Gravity Counter: {self.gravity_counter}"

    def valid_moves(self):
        valid_actions = []
        for action in range(4):
            # Map action to movement
            if action == 0:
                move = 1
            elif action == 1:
                move = 2
            elif action == 2:
                move = -1
            elif action == 3:
                move = -2
            new_counter = self.gravity_counter + move
            if -10 <= new_counter <= 10:
                valid_actions.append(action)
        return valid_actions
