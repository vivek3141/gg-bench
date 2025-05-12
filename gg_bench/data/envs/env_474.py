import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 = Add 1, 1 = Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Define observation space: Current number between 1 and 20
        self.observation_space = spaces.Box(low=1, high=20, shape=(1,), dtype=np.int32)

        # Initialize state
        self.current_number = None
        self.current_player = None
        self.done = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return (np.array([self.current_number], dtype=np.int32), 0, True, False, {})

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action, player loses
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply the action
        if action == 0:
            next_number = self.current_number + 1
        elif action == 1:
            next_number = self.current_number * 2

        # Update current number
        self.current_number = next_number

        # Check for win condition
        if self.current_number == 20:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if the move exceeded the target number
        if self.current_number > 20:
            # Current player loses
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1

        # Check if the next player has valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Next player has no valid moves, switch back to current player
            self.current_player *= -1

            # Check if the current player has valid moves
            valid_actions = self.valid_moves()
            if not valid_actions:
                # No valid moves for either player, game ends in a draw
                self.done = True
                reward = 0
                return (
                    np.array([self.current_number], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )

        # Continue the game
        reward = 0
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return (
            f"Current Number: {self.current_number}\n"
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )

    def valid_moves(self):
        valid = []
        if self.current_number + 1 <= 20:
            valid.append(0)  # Add 1
        if self.current_number * 2 <= 20:
            valid.append(1)  # Multiply by 2
        return valid
