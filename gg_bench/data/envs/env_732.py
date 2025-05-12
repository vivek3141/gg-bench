import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.starting_number = 60
        self.max_number = 60
        self.action_space = spaces.Discrete(
            self.max_number + 1
        )  # Actions from 0 to 60 inclusive
        self.observation_space = spaces.Box(
            low=1, high=self.max_number, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array([self.current_number], dtype=np.int32)
        return observation, {}  # observation, info

    def valid_moves(self):
        n = self.current_number
        # Proper divisors exclude 1 and the number itself
        divisors = [i for i in range(2, n) if n % i == 0]
        return divisors

    def step(self, action):
        if self.done:
            # Game is already over
            observation = np.array([self.current_number], dtype=np.int32)
            return observation, 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            observation = np.array([self.current_number], dtype=np.int32)
            return observation, reward, True, False, {}

        # Apply the valid action
        self.current_number -= action

        # Check for win conditions
        if self.current_number == 1:
            # Current player wins by reducing the number to 1
            self.done = True
            reward = 1
            observation = np.array([self.current_number], dtype=np.int32)
            return observation, reward, True, False, {}
        else:
            # Check if the next player has valid moves
            next_valid_moves = self.valid_moves()
            if not next_valid_moves:
                # Next player cannot move; current player wins
                self.done = True
                reward = 1
                observation = np.array([self.current_number], dtype=np.int32)
                return observation, reward, True, False, {}
            else:
                # Switch to the next player
                self.current_player *= -1
                reward = 0
                observation = np.array([self.current_number], dtype=np.int32)
                return observation, reward, False, False, {}

    def render(self):
        current_player = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current Number: {self.current_number}, {current_player}'s turn."
