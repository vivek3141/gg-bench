import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_number=100):
        super(CustomEnv, self).__init__()
        self.target_number = target_number

        # Define action and observation space
        # Two possible actions: 0 (Add 1), 1 (Multiply by 2)
        self.action_space = spaces.Discrete(2)

        # Observation space includes current player's number and opponent's number
        self.observation_space = spaces.Box(
            low=1, high=self.target_number, shape=(2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_numbers = [1, 1]  # Both players start with 1
        self.current_player = 0  # Player 0 starts
        self.done = False
        return (
            np.array(self.current_numbers, dtype=np.float32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(self.current_numbers, dtype=np.float32),
                0,
                True,
                False,
                {},
            )

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            reward = -10
            return (
                np.array(self.current_numbers, dtype=np.float32),
                reward,
                True,
                False,
                {},
            )

        # Get current player's number
        current_number = self.current_numbers[self.current_player]

        # Apply action
        if action == 0:
            new_number = current_number + 1
        elif action == 1:
            new_number = current_number * 2

        # Update current player's number
        self.current_numbers[self.current_player] = new_number

        # Check for win condition
        if new_number == self.target_number:
            self.done = True
            reward = 1
            return (
                np.array(self.current_numbers, dtype=np.float32),
                reward,
                True,
                False,
                {},
            )

        # Since action is valid, new_number cannot exceed target_number here

        # Switch to the next player
        self.current_player = 1 - self.current_player

        return (np.array(self.current_numbers, dtype=np.float32), 0, False, False, {})

    def render(self):
        player_strings = []
        for i in range(2):
            player_str = f"Player {i + 1}: Current Number = {self.current_numbers[i]}"
            if self.current_player == i and not self.done:
                player_str += " (Current Turn)"
            player_strings.append(player_str)
        return "\n".join(player_strings)

    def valid_moves(self):
        # Returns a list of valid action indices that do not exceed target number
        current_number = self.current_numbers[self.current_player]
        valid_actions = []
        # Check Add 1
        if current_number + 1 <= self.target_number:
            valid_actions.append(0)
        # Check Multiply by 2
        if current_number * 2 <= self.target_number:
            valid_actions.append(1)
        return valid_actions
