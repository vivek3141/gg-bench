import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: integers from -10 to -1 and 1 to 10 (excluding 0), mapped to indices 0-19
        self.action_space = spaces.Discrete(20)

        # Observation space: Current balance between -20 and 20
        self.observation_space = spaces.Box(
            low=-20, high=20, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_balance = 0  # Starting balance
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_balance], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Map action index to move value
        move = self._action_to_move(action)

        # Check if the move is valid (move is between -10 to -1 or 1 to 10)
        if move == 0:
            # Should not occur, but included for safety
            return (
                np.array([self.current_balance], dtype=np.int32),
                0,
                True,
                False,
                {"error": "Invalid move: 0 is not allowed"},
            )

        # Update the balance
        self.current_balance += move

        # Check for win/loss condition
        if self.current_balance < -20 or self.current_balance > 20:
            # Current player loses
            self.done = True
            reward = 0  # Loss
            return (
                np.array([self.current_balance], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Game continues
        reward = -10  # Valid move
        return (
            np.array([self.current_balance], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        balance_str = f"Current balance: {self.current_balance}\n"
        balance_str += f"Player {self.current_player}'s turn."
        return balance_str

    def valid_moves(self):
        # Returns a list of valid action indices based on current balance
        valid_actions = []
        moves = list(range(-10, 0)) + list(range(1, 11))  # Possible moves

        for idx, move in enumerate(moves):
            new_balance = self.current_balance + move
            if -20 <= new_balance <= 20:
                valid_actions.append(idx)

        return valid_actions

    def _action_to_move(self, action):
        # Map action index to actual move value (-10 to -1, 1 to 10)
        if 0 <= action <= 9:
            return action - 10  # Maps indices 0-9 to moves -10 to -1
        elif 10 <= action <= 19:
            return action - 9  # Maps indices 10-19 to moves 1 to 10
