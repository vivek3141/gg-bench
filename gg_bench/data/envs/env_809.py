import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=15):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number

        # Define action and observation space
        # Two possible actions: 0 - Subtract 1, 1 - Divide by 2
        self.action_space = spaces.Discrete(2)
        # Observation space: [shared_number, current_player]
        # shared_number ranges from 0 to a maximum int32 value
        # current_player is 1 or 2
        self.observation_space = spaces.Box(
            low=np.array([0, 1], dtype=np.int32),
            high=np.array([np.iinfo(np.int32).max, 2], dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array(
            [self.shared_number, self.current_player], dtype=np.int32
        )
        info = {}
        return observation, info  # Return observation and info

    def step(self, action):
        # Check if the game has already ended
        if self.done:
            observation = np.array(
                [self.shared_number, self.current_player], dtype=np.int32
            )
            return observation, 0, True, False, {}

        # Check if the shared number is zero at the start of the turn
        if self.shared_number == 0:
            # Current player cannot make a move and loses
            reward = -10
            self.done = True
            observation = np.array(
                [self.shared_number, self.current_player], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Determine valid actions
        valid_actions = [0]  # Subtract 1 is always valid when shared_number > 0
        if self.shared_number % 2 == 0:
            valid_actions.append(1)  # Divide by 2 is valid when shared_number is even

        # Check if the action is valid
        if action not in valid_actions:
            # Invalid action
            reward = -10
            self.done = True
            observation = np.array(
                [self.shared_number, self.current_player], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Perform the action
        if action == 0:
            # Subtract 1
            self.shared_number -= 1
        elif action == 1:
            # Divide by 2
            self.shared_number = self.shared_number // 2

        # Check if the player wins
        if self.shared_number == 0:
            # Current player wins
            reward = 1
            self.done = True
            observation = np.array(
                [self.shared_number, self.current_player], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        observation = np.array(
            [self.shared_number, self.current_player], dtype=np.int32
        )
        return observation, reward, False, False, {}

    def render(self):
        return (
            f"Player {self.current_player}'s Turn. Shared Number: {self.shared_number}"
        )

    def valid_moves(self):
        # Determine valid actions based on the current shared number
        if self.shared_number == 0:
            return []
        valid_actions = [0]  # Subtract 1 is always valid when shared_number > 0
        if self.shared_number % 2 == 0:
            valid_actions.append(1)  # Divide by 2 is valid when shared_number is even
        return valid_actions
