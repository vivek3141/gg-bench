import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=1, target_number=20):
        super(CustomEnv, self).__init__()
        # Game parameters
        self.starting_number = starting_number
        self.target_number = target_number

        # Define action space: 0 = Add 1, 1 = Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Define observation space: Current number can range from 1 to target_number * 2
        self.observation_space = spaces.Box(
            low=1, high=target_number * 2, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None:
            # Update starting and target numbers if provided in options
            self.starting_number = options.get("starting_number", self.starting_number)
            self.target_number = options.get("target_number", self.target_number)
            # Update observation space if target number has changed
            self.observation_space = spaces.Box(
                low=1, high=self.target_number * 2, shape=(1,), dtype=np.int32
            )
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # Game is over; no more actions can be taken
            return np.array([self.current_number], dtype=np.int32), 0, self.done, {}

        # Apply the selected action
        if action == 0:
            self.current_number += 1  # Add 1
        elif action == 1:
            self.current_number *= 2  # Multiply by 2
        else:
            # Invalid action (should not occur as action_space is Discrete(2))
            raise ValueError("Invalid action")

        # Check for winning condition
        if self.current_number == self.target_number:
            reward = 1  # Current player wins
            self.done = True
        elif self.current_number > self.target_number:
            reward = -10  # Current player loses for exceeding target number
            self.done = True
        else:
            reward = 0  # No win or loss; game continues
            # Switch to the other player
            self.current_player = 3 - self.current_player

        return np.array([self.current_number], dtype=np.int32), reward, self.done, {}

    def render(self):
        # Provide a simple text representation of the game state
        return f"Current Number: {self.current_number}, Player Turn: Player {self.current_player}"

    def valid_moves(self):
        # Determine valid actions based on the current number and target number
        valid_actions = []
        # Check if adding 1 stays within the target
        if self.current_number + 1 <= self.target_number:
            valid_actions.append(0)
        # Check if multiplying by 2 stays within the target
        if self.current_number * 2 <= self.target_number:
            valid_actions.append(1)
        return valid_actions
