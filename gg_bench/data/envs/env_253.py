import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: movements of 1, 2, or 3 steps (encoded as 0, 1, 2)
        self.action_space = spaces.Discrete(3)

        # Observation space: [position, current_player]
        # position: integer from 0 to 20
        # current_player: 1 or 2
        self.observation_space = spaces.Box(
            low=np.array([0, 1]), high=np.array([20, 2]), dtype=np.int32
        )

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Token starts at position 10
        self.position = 10
        # Player 1 starts first
        self.current_player = 1
        self.done = False
        return np.array([self.position, self.current_player], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is over, no more actions are valid
            return (
                np.array([self.position, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if action not in [0, 1, 2]:
            # Invalid action
            self.done = True
            return (
                np.array([self.position, self.current_player], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Map action to movement value (1, 2, or 3)
        movement = action + 1

        # Determine new position based on current player
        if self.current_player == 1:
            # Player 1 moves left towards position 0
            new_position = self.position - movement
            if new_position < 0:
                # Invalid move beyond the number line
                self.done = True
                return (
                    np.array([self.position, self.current_player], dtype=np.int32),
                    -10,
                    True,
                    False,
                    {},
                )
        elif self.current_player == 2:
            # Player 2 moves right towards position 20
            new_position = self.position + movement
            if new_position > 20:
                # Invalid move beyond the number line
                self.done = True
                return (
                    np.array([self.position, self.current_player], dtype=np.int32),
                    -10,
                    True,
                    False,
                    {},
                )
        else:
            # Invalid current player value
            self.done = True
            return (
                np.array([self.position, self.current_player], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Update position
        self.position = new_position

        # Check for win condition
        if self.current_player == 1 and self.position == 0:
            # Player 1 wins
            self.done = True
            return (
                np.array([self.position, self.current_player], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        elif self.current_player == 2 and self.position == 20:
            # Player 2 wins
            self.done = True
            return (
                np.array([self.position, self.current_player], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        else:
            # Switch to the other player
            self.current_player = 2 if self.current_player == 1 else 1
            return (
                np.array([self.position, self.current_player], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        # Return a string representation of the current game state
        number_line = ["-" for _ in range(21)]
        number_line[self.position] = "T"  # Mark the token's position
        number_line_str = "".join(number_line)
        return f"Player {self.current_player}'s turn.\nPosition: {self.position}\n{number_line_str}"

    def valid_moves(self):
        # Return a list of valid action indices (0, 1, 2)
        valid_actions = []
        for action in [0, 1, 2]:
            movement = action + 1
            if self.current_player == 1:
                # Check if moving left is valid
                new_position = self.position - movement
                if new_position >= 0:
                    valid_actions.append(action)
            elif self.current_player == 2:
                # Check if moving right is valid
                new_position = self.position + movement
                if new_position <= 20:
                    valid_actions.append(action)
        return valid_actions
