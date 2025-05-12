import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.target_number = 23  # Fixed target number
        self.action_space = spaces.Discrete(3)  # Actions: 0:+2, 1:*2, 2:-1
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.target_number]), dtype=np.int32
        )
        self.current_number = None
        self.current_player = None
        self.done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return np.array([self.current_number], dtype=np.int32), -10, True, False, {}

        # Map action to operation
        if action == 0:
            new_current_number = self.current_number + 2
        elif action == 1:
            new_current_number = self.current_number * 2
        elif action == 2:
            new_current_number = self.current_number - 1
        else:
            # Should not happen since we checked action in valid_moves()
            raise ValueError(f"Invalid action: {action}")

        # Update current number
        self.current_number = new_current_number

        # Check for win/loss conditions
        if self.current_number == self.target_number:
            reward = 1  # Current player wins
            self.done = True
            terminated = True
        elif self.current_number > self.target_number:
            reward = -10  # Current player loses
            self.done = True
            terminated = True
        else:
            reward = 0
            terminated = False
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1

        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            terminated,
            False,
            {},
        )

    def render(self):
        return f"Current Player: Player {self.current_player}, Current Number: {self.current_number}, Target Number: {self.target_number}"

    def valid_moves(self):
        valid_actions = [0, 1]  # +2 and *2 are always valid
        if self.current_number >= 2:
            valid_actions.append(2)  # -1 is valid only if current number >= 2
        return valid_actions
