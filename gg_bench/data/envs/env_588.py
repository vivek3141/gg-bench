import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=100):
        super(CustomEnv, self).__init__()

        self.N = N  # Target number

        # Define action space
        # Actions 0-8: Add 1-9
        # Actions 9-12: Multiply by 2-5
        self.action_space = spaces.Discrete(13)

        # Observation space: current number (integer between 1 and N)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.N]), dtype=np.int32
        )

        # Initialize the game state
        self.current_number = None
        self.current_player = None
        self.done = None

        # Seed the environment (optional)
        self.seed()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array([self.current_number], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # The game is over; no more moves can be made
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Perform the action
        if 0 <= action <= 8:
            # Addition
            add_amount = action + 1  # Map action to add amount (1-9)
            new_number = self.current_number + add_amount
        elif 9 <= action <= 12:
            # Multiplication
            multiply_by = action - 7  # Map action to multiplier (2-5)
            new_number = self.current_number * multiply_by
        else:
            # This should not happen due to valid_moves check
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if the new number exceeds the target
        if new_number > self.N:
            # Invalid move (should not happen due to valid_moves)
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Update the current number
        self.current_number = new_number

        # Check for win condition
        if self.current_number == self.N:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if any valid moves are left
        if len(self.valid_moves()) == 0:
            self.done = True
            reward = -10  # Current player loses
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1
        reward = 0  # No reward yet
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}\n"

    def valid_moves(self):
        valid_actions = []

        # Check possible addition moves
        for action in range(0, 9):
            add_amount = action + 1  # Actions 0-8 correspond to add 1-9
            new_number = self.current_number + add_amount
            if new_number <= self.N:
                valid_actions.append(action)

        # Check possible multiplication moves
        for action in range(9, 13):
            multiply_by = action - 7  # Actions 9-12 correspond to multiply by 2-5
            new_number = self.current_number * multiply_by
            if new_number <= self.N:
                valid_actions.append(action)

        return valid_actions
