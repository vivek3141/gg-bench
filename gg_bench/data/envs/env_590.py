import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space and observation space
        # Action space: 10 discrete actions (add/multiply with numbers 1-5)
        self.action_space = spaces.Discrete(10)

        # Observation space: Current number, range from 1 to 1000
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([1000]), shape=(1,), dtype=np.int32
        )

        # Precompute the action map
        self.action_map = []
        for op in ["add", "multiply"]:
            for num in [1, 2, 3, 4, 5]:
                self.action_map.append((op, num))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        if action not in range(self.action_space.n):
            self.done = True
            reward = -10  # Invalid action
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        operation, number = self.action_map[action]

        # Check if action is valid (resulting number does not exceed 1000)
        temp_number = self.current_number
        if operation == "add":
            temp_number += number
        elif operation == "multiply":
            temp_number *= number

        if temp_number > 1000:
            self.done = True
            reward = -10  # Invalid move
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply the action
        self.current_number = temp_number

        # Check for win condition
        if self.current_number >= 31:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0  # No reward if the game continues
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        render_str = f"Current Number: {self.current_number}\nPlayer {self.current_player}'s turn."
        return render_str

    def valid_moves(self):
        valid_actions = []
        for idx, (operation, number) in enumerate(self.action_map):
            temp_number = self.current_number
            if operation == "add":
                temp_number += number
            elif operation == "multiply":
                temp_number *= number
            if temp_number <= 1000:
                valid_actions.append(idx)
        return valid_actions
