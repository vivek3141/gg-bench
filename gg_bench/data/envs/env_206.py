import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum initial total T
        self.initial_T = 30  # You can choose any composite number greater than 1
        self.max_T = self.initial_T

        # Define action and observation space
        # Action space corresponds to integers from 0 to max_T (actions correspond to possible divisors to subtract)
        self.action_space = spaces.Discrete(self.max_T + 1)

        # Observation space is the current total T
        self.observation_space = spaces.Box(
            low=1, high=self.max_T, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.T = self.initial_T  # Current total T
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.T], dtype=np.int32), {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return (
                np.array([self.T], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Check if there are valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player cannot move; they lose
            self.done = True
            reward = -1
            return np.array([self.T], dtype=np.int32), reward, True, False, {}

        # Check if the action is valid
        if action not in valid_moves:
            # Invalid move; current player loses
            self.done = True
            reward = -10
            return np.array([self.T], dtype=np.int32), reward, True, False, {}

        # Valid move; subtract the divisor from T
        self.T -= action

        # Check if the current player wins
        if self.T == 1:
            self.done = True
            reward = 1
            return np.array([self.T], dtype=np.int32), reward, True, False, {}

        # Current player made a valid move but did not win
        reward = -10  # As per the prompt, valid move gets a reward of -10

        # Now it's the opponent's turn
        # Check if the opponent can make a valid move
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot move; current player wins
            self.done = True
            reward = 1
            return np.array([self.T], dtype=np.int32), reward, True, False, {}

        # Opponent makes a move (you can implement a strategy; here we choose randomly)
        opponent_action = np.random.choice(opponent_valid_moves)
        self.T -= opponent_action

        # Check if the opponent wins
        if self.T == 1:
            self.done = True
            reward = -1  # Current player loses
            return np.array([self.T], dtype=np.int32), reward, True, False, {}

        # Return the observation after opponent's move
        return np.array([self.T], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current total (T): {self.T}"

    def valid_moves(self):
        # Proper divisors of T (excluding 1 and T itself)
        if self.T <= 3:
            return []
        proper_divisors = [i for i in range(2, self.T) if self.T % i == 0]
        return proper_divisors
