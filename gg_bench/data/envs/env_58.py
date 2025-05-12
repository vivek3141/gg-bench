import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_N=100):
        super(CustomEnv, self).__init__()

        # Initial starting number N, can be parameterized
        self.starting_N = starting_N

        # Maximum value of N we expect
        self.max_N = self.starting_N

        # Define action and observation space
        # Actions correspond to possible divisors from 2 to max_N
        # Since action_space must be Discrete, and action_space.n = number of actions

        # We set action_space to Discrete(max_N + 1)
        # action = integer between 0 and max_N
        self.action_space = spaces.Discrete(self.max_N + 1)

        # Observation space is the current value of N
        # It's a one-element array, values from 1 to max_N
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.max_N]), dtype=np.int32
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_N
        self.done = False
        self.current_player = 1  # 1: Player 1; -1: Player 2
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Environment already done
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        # Action corresponds to a number between 0 and max_N
        # Valid moves are in valid_divisors
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player cannot move at the start of their turn
            # This should not happen here, as the previous player should have won
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        if action not in valid_moves:
            # Invalid move
            self.done = True
            reward = -10  # As per specification
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Apply the action
        self.N = self.N // action  # Update N

        # Check if N is now 1
        if self.N == 1:
            # Current player wins
            self.done = True
            reward = 1  # Current player wins
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Switch to next player
        self.current_player *= -1

        # Check if next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Next player cannot move, current player wins
            self.done = True
            # Since current_player has already been switched, switch back to give reward to correct player
            self.current_player *= -1  # Switch back
            reward = 1  # Current player wins
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Game continues
        reward = 0
        return (
            np.array([self.N], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Return a string representation
        state_str = f"Current N: {self.N}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        valid_moves = self.valid_moves()
        state_str += f"Valid moves: {valid_moves}\n"
        return state_str

    def valid_moves(self):
        # Return list of valid moves (divisors >=2 and <N)
        if self.N <= 1:
            return []
        valid_divisors = [d for d in range(2, self.N) if self.N % d == 0]
        return valid_divisors
