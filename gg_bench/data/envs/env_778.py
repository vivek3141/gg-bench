import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Move 1 step left
        #          1 = Move 1 step right
        #          2 = Move 2 steps left
        #          3 = Move 2 steps right
        self.action_space = spaces.Discrete(4)

        # Observation space: Token position on the number line (positions 1 to 10)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([10]), shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting position of the token is 5
        self.token_position = 5
        self.current_player = 1  # Can be 1 or -1
        self.done = False
        return np.array([self.token_position], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.token_position], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Map action to movement
        action_map = {
            0: (1, "left"),
            1: (1, "right"),
            2: (2, "left"),
            3: (2, "right"),
        }

        if action not in self.action_space:
            # Invalid action
            self.done = True
            return (
                np.array([self.token_position], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        positions_to_move, direction = action_map[action]

        # Calculate new position
        if direction == "left":
            new_position = self.token_position - positions_to_move
        elif direction == "right":
            new_position = self.token_position + positions_to_move
        else:
            # Invalid direction (should not occur)
            self.done = True
            return (
                np.array([self.token_position], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Validate move
        if new_position < 1 or new_position > 10:
            # Invalid move
            self.done = True
            return (
                np.array([self.token_position], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Update token position
        self.token_position = new_position

        # Check for loss condition
        if self.token_position == 1 or self.token_position == 10:
            # Current player loses
            self.done = True
            return (
                np.array([self.token_position], dtype=np.int32),
                -1,
                True,
                False,
                {},
            )

        # Switch current player
        self.current_player *= -1

        return (
            np.array([self.token_position], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        # Visual representation of the number line and token position
        line = "Positions: "
        for i in range(1, 11):
            if i == self.token_position:
                line += f"[{i}] "
            else:
                line += f" {i}  "
        return line

    def valid_moves(self):
        # Returns a list of valid action indices
        valid_actions = []
        action_map = {
            0: (1, "left"),
            1: (1, "right"),
            2: (2, "left"),
            3: (2, "right"),
        }
        for action in range(4):
            positions_to_move, direction = action_map[action]
            if direction == "left":
                new_position = self.token_position - positions_to_move
            else:
                new_position = self.token_position + positions_to_move
            if 1 <= new_position <= 10:
                valid_actions.append(action)
        return valid_actions
