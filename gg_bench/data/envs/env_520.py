import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: indices correspond to square numbers 1, 4, 9, 16
        self.square_numbers = [1, 4, 9, 16]
        self.action_space = spaces.Discrete(len(self.square_numbers))

        # Define observation space: [current_total, current_player]
        # current_total ranges from 0 to 20
        # current_player is -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([0, -1]), high=np.array([20, 1]), shape=(2,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = 20
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_total, self.current_player], dtype=np.int32),
            {},
        )  # Observation, info

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_total, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Map action to square number
        try:
            square_num = self.square_numbers[action]
        except IndexError:
            # Invalid action index
            self.done = True
            return (
                np.array([self.current_total, self.current_player], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        if square_num > self.current_total:
            # Invalid move
            self.done = True
            return (
                np.array([self.current_total, self.current_player], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Valid move
        self.current_total -= square_num

        if self.current_total == 0:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_total, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        else:
            # Continue game
            reward = 0
            self.current_player *= -1  # Switch player
            return (
                np.array([self.current_total, self.current_player], dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        available_squares = [
            str(num) for num in self.square_numbers if num <= self.current_total
        ]
        render_str = (
            f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
            f"Current Total: {self.current_total}\n"
            f"Available squares to subtract: {', '.join(available_squares)}\n"
        )
        return render_str

    def valid_moves(self):
        return [
            idx
            for idx, num in enumerate(self.square_numbers)
            if num <= self.current_total
        ]
