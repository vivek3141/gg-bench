import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 18 possible actions (numbers 1-9, each with add or subtract)
        self.action_space = spaces.Discrete(18)

        # Define observation space:
        # Observation is a vector of 10 elements:
        # [shared_total, number1_status, number2_status, ..., number9_status]
        # shared_total ranges from -45 to +45
        # number statuses are 0 (used) or 1 (available)
        self.observation_space = spaces.Box(
            low=np.array([-45] + [0] * 9, dtype=np.int32),
            high=np.array([45] + [1] * 9, dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_total = 0
        self.number_pool = [1] * 9  # 1 for available, 0 for used
        self.current_player = 1  # Players are 1 and 2
        self.done = False
        observation = np.array([self.shared_total] + self.number_pool, dtype=np.int32)
        return observation, {}

    def step(self, action):
        if self.done:
            observation = np.array(
                [self.shared_total] + self.number_pool, dtype=np.int32
            )
            return observation, 0, True, False, {}

        # Parse action into number and operation
        number = action // 2 + 1  # numbers from 1 to 9
        op = action % 2  # 0 for Add, 1 for Subtract

        # Check if number is available
        if self.number_pool[number - 1] == 0:
            # Invalid move: number already used
            self.done = True
            reward = -10
            terminated = True
            observation = np.array(
                [self.shared_total] + self.number_pool, dtype=np.int32
            )
            return observation, reward, terminated, False, {}

        # Apply operation
        if op == 0:
            # Add
            self.shared_total += number
        else:
            # Subtract
            self.shared_total -= number

        # Mark number as used
        self.number_pool[number - 1] = 0

        # Check for win
        if self.shared_total == 0:
            # Current player wins
            self.done = True
            reward = 1
            terminated = True
            observation = np.array(
                [self.shared_total] + self.number_pool, dtype=np.int32
            )
            return observation, reward, terminated, False, {}

        # Check if all numbers are used
        if not any(self.number_pool):
            # All numbers used
            # According to rules, current player wins if shared_total != 0
            self.done = True
            reward = 1
            terminated = True
            observation = np.array(
                [self.shared_total] + self.number_pool, dtype=np.int32
            )
            return observation, reward, terminated, False, {}

        # Game continues
        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        terminated = False
        observation = np.array([self.shared_total] + self.number_pool, dtype=np.int32)
        return observation, reward, terminated, False, {}

    def render(self):
        available_numbers = [str(i + 1) for i in range(9) if self.number_pool[i] == 1]
        state_str = f"Player {self.current_player}'s turn\n"
        state_str += f"Available Numbers: {', '.join(available_numbers)}\n"
        state_str += f"Shared Total: {self.shared_total}\n"
        return state_str

    def valid_moves(self):
        valid_actions = []
        for number in range(1, 10):
            if self.number_pool[number - 1] == 1:
                action_add = (number - 1) * 2  # Add action index
                action_subtract = (number - 1) * 2 + 1  # Subtract action index
                valid_actions.extend([action_add, action_subtract])
        return valid_actions
