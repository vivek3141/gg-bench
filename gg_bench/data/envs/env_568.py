import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Default starting number
        self.initial_number = 30

        # Define action and observation space
        self.action_space = spaces.Discrete(
            self.initial_number + 1
        )  # Actions from 0 to initial_number
        self.observation_space = spaces.Box(
            low=0, high=self.initial_number, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = self.initial_number
        self.done = False
        return (
            np.array([self.shared_number], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.shared_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Check for valid moves for the agent
        agent_valid_moves = self.get_proper_divisors(self.shared_number)
        if not agent_valid_moves:
            # Agent loses
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                -1,
                True,
                False,
                {},
            )

        # Check if the action is valid
        if action not in agent_valid_moves:
            # Invalid move
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Agent makes a valid move
        self.shared_number -= action

        if self.shared_number == 0:
            # Agent wins
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Opponent's turn
        opponent_valid_moves = self.get_proper_divisors(self.shared_number)
        if not opponent_valid_moves:
            # Opponent loses, agent wins
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Opponent makes a random valid move
        opponent_action = random.choice(opponent_valid_moves)
        self.shared_number -= opponent_action

        if self.shared_number == 0:
            # Opponent wins, agent loses
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                -1,
                True,
                False,
                {},
            )

        # Game continues
        return (
            np.array([self.shared_number], dtype=np.int32),
            -10,
            False,
            False,
            {},
        )

    def render(self):
        return f"Shared Number: {self.shared_number}"

    def valid_moves(self):
        return self.get_proper_divisors(self.shared_number)

    def get_proper_divisors(self, n):
        if n <= 1:
            return []
        divisors = []
        for i in range(2, n):
            if n % i == 0:
                divisors.append(i)
        return divisors
