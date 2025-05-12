import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to multipliers from 2 to 9 inclusive (actions 0 to 7)
        self.action_space = spaces.Discrete(8)
        # Observation is the current number, an integer between 1 and 100
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([100]), dtype=np.int32
        )

        # Initialize environment state
        self.current_number = None
        self.current_player = None
        self.terminated = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1  # Starting number
        self.current_player = 1  # Player 1 starts
        self.terminated = False
        return np.array([self.current_number], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        # Map action to multiplier (actions 0-7 map to multipliers 2-9)
        multiplier = action + 2

        # Check if the game is already over
        if self.terminated:
            return (
                np.array([self.current_number], dtype=np.int32),
                0.0,
                True,
                False,
                {},
            )

        # Validate action
        if multiplier < 2 or multiplier > 9:
            # Invalid multiplier selected
            self.terminated = True
            reward = -10.0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Calculate new current number
        new_number = self.current_number * multiplier
        if new_number > 100:
            # Exceeded target number; player loses
            self.terminated = True
            reward = -10.0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        self.current_number = new_number

        # Check for winning condition
        if self.current_number == 100:
            # Player wins by reaching exactly 100
            self.terminated = True
            reward = 1.0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1

        # Check if the next player has any valid moves
        if len(self.valid_moves()) == 0:
            # Next player has no valid moves; current player wins
            self.terminated = True
            # Switch back to current player to assign reward correctly
            self.current_player *= -1
            reward = 1.0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Game continues
        reward = 0.0
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        # Provide a string representation of the current state
        status = f"Current Number: {self.current_number}\n"
        status += f"Player {'1' if self.current_player == 1 else '2'}'s turn."
        return status

    def valid_moves(self):
        # Return a list of valid action indices based on the current number
        valid_actions = []
        for action in range(8):  # Actions correspond to multipliers 2 to 9
            multiplier = action + 2
            if self.current_number * multiplier <= 100:
                valid_actions.append(action)
        return valid_actions
