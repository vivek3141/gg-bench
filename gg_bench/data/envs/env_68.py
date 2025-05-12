import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=60):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number

        # Define action and observation space
        # Action space: integers from 0 to starting_number inclusive
        # The agent's action is the proper divisor to subtract from the current number
        self.action_space = spaces.Discrete(self.starting_number + 1)
        self.observation_space = spaces.Box(
            low=np.array([2, -1]),
            high=np.array([self.starting_number, 1]),
            dtype=np.int32,
        )

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return np.array([self.current_number, self.current_player], dtype=np.int32), {}

    def get_proper_divisors(self, n):
        # Returns a list of proper divisors of n, excluding 1 and n itself
        divisors = [i for i in range(2, n) if n % i == 0]
        return divisors

    def valid_moves(self):
        # Returns a list of valid moves (proper divisors of current_number)
        return self.get_proper_divisors(self.current_number)

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Valid move: subtract action from current_number
        self.current_number -= action

        # Check if opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot make a move; current player wins
            self.done = True
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        else:
            # Switch to opponent's turn
            self.current_player *= -1
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        current_player_str = "Player 1" if self.current_player == 1 else "Player 2"
        render_str = f"{current_player_str}'s turn.\n"
        render_str += f"Current Number: {self.current_number}\n"
        render_str += f"Proper Divisors: {self.valid_moves()}\n"
        return render_str
