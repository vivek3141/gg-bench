import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: digits from 1 to 9 (actions 0 to 8)
        self.action_space = spaces.Discrete(9)  # Actions correspond to digits 1-9

        # Define observation space: remainders modulo 2, 3, 5, and 7
        # Remainders range from 0 to modulus - 1
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, 2, 4, 6]),
            dtype=np.int32,
        )

        # Forbidden divisors and target divisor
        self.forbidden_numbers = [2, 3, 5]
        self.target_number = 7

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize shared number and remainders
        self.shared_number = 0
        self.remainders = {2: 0, 3: 0, 5: 0, 7: 0}
        self.done = False

        # Return observation and info
        observation = np.array(
            [
                self.remainders[2],
                self.remainders[3],
                self.remainders[5],
                self.remainders[7],
            ],
            dtype=np.int32,
        )
        return observation, {}

    def step(self, action):
        if self.done:
            # If the game is already over
            observation = np.array(
                [
                    self.remainders[2],
                    self.remainders[3],
                    self.remainders[5],
                    self.remainders[7],
                ],
                dtype=np.int32,
            )
            return observation, 0, True, False, {}

        # Check for valid moves
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # No valid moves available; current player loses
            self.done = True
            reward = -10
            observation = np.array(
                [
                    self.remainders[2],
                    self.remainders[3],
                    self.remainders[5],
                    self.remainders[7],
                ],
                dtype=np.int32,
            )
            return observation, reward, True, False, {}

        if action not in valid_actions:
            # Invalid move; current player loses
            self.done = True
            reward = -10
            observation = np.array(
                [
                    self.remainders[2],
                    self.remainders[3],
                    self.remainders[5],
                    self.remainders[7],
                ],
                dtype=np.int32,
            )
            return observation, reward, True, False, {}

        # Map action to digit (action 0-8 corresponds to digit 1-9)
        digit = action + 1

        # Update shared number and remainders
        self.shared_number = self.shared_number * 10 + digit
        for m in [2, 3, 5, 7]:
            self.remainders[m] = (self.remainders[m] * 10 + digit % m) % m

        # Check for forbidden divisibility
        for m in self.forbidden_numbers:
            if self.remainders[m] == 0:
                # Divisible by forbidden number; current player loses
                self.done = True
                reward = -10
                observation = np.array(
                    [
                        self.remainders[2],
                        self.remainders[3],
                        self.remainders[5],
                        self.remainders[7],
                    ],
                    dtype=np.int32,
                )
                return observation, reward, True, False, {}

        # Check for winning condition
        if self.remainders[self.target_number] == 0:
            # Divisible by target number; current player wins
            self.done = True
            reward = 1
            observation = np.array(
                [
                    self.remainders[2],
                    self.remainders[3],
                    self.remainders[5],
                    self.remainders[7],
                ],
                dtype=np.int32,
            )
            return observation, reward, True, False, {}

        # Valid move; game continues without reward
        reward = 0
        observation = np.array(
            [
                self.remainders[2],
                self.remainders[3],
                self.remainders[5],
                self.remainders[7],
            ],
            dtype=np.int32,
        )
        return observation, reward, False, False, {}

    def render(self):
        # Return a string visualizing the current shared number
        return f"Current Shared Number: {self.shared_number}"

    def valid_moves(self):
        # Compute and return a list of valid actions (0-8)
        valid_actions = []
        for action in range(9):
            digit = action + 1
            is_valid = True
            for m in self.forbidden_numbers:
                remainder = (self.remainders[m] * 10 + digit % m) % m
                if remainder == 0:
                    is_valid = False
                    break
            if is_valid:
                valid_actions.append(action)
        return valid_actions
