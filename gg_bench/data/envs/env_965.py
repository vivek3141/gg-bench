import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_number=23):
        super(CustomEnv, self).__init__()

        self.target_number = target_number

        # Define action and observation space
        # Actions: 0 - Add 1, 1 - Add 2, 2 - Multiply by 2
        self.action_space = spaces.Discrete(3)

        # Observation space: The current number (integer between 1 and target_number)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.target_number]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is already over, no further moves are allowed
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move results in a penalty and ends the episode
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Apply the action
        if action == 0:
            new_number = self.current_number + 1
        elif action == 1:
            new_number = self.current_number + 2
        elif action == 2:
            new_number = self.current_number * 2

        self.current_number = new_number

        # Check for victory
        if self.current_number == self.target_number:
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Check if the current number exceeds the target (should not happen due to valid_moves)
        if self.current_number > self.target_number:
            # This should not occur if valid_moves is implemented correctly
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Switch to the other player
        self.current_player *= -1

        return np.array([self.current_number], dtype=np.int32), 0, False, False, {}

    def render(self):
        state_str = f"Current Number: {self.current_number}, Target Number: {self.target_number}"
        return state_str

    def valid_moves(self):
        # Return a list of valid actions based on the current number
        actions = []
        # Action 0: Add 1
        if self.current_number + 1 <= self.target_number:
            actions.append(0)
        # Action 1: Add 2
        if self.current_number + 2 <= self.target_number:
            actions.append(1)
        # Action 2: Multiply by 2
        if self.current_number * 2 <= self.target_number:
            actions.append(2)
        return actions
