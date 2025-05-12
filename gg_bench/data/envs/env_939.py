import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=23):
        super(CustomEnv, self).__init__()

        # Initial shared countdown number
        self.initial_N = N
        self.N = N

        # Allowed prime numbers to subtract
        self.allowed_primes = [2, 3, 5, 7]

        # Action space: indices of allowed primes
        self.action_space = spaces.Discrete(len(self.allowed_primes))

        # Observation space: current value of N
        self.observation_space = spaces.Box(
            low=0, high=self.initial_N, shape=(1,), dtype=np.int32
        )

        # Current player: 1 or -1 (Player 1 or Player 2)
        self.current_player = 1

        # Game over flag
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the game state
        self.N = self.initial_N
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            return (np.array([self.N], dtype=np.int32), 0, True, False, {})

        prime_to_subtract = self.allowed_primes[action]

        # Check if the move is valid
        if prime_to_subtract > self.N:
            # Invalid move: subtracting too large a prime
            self.done = True
            reward = -10  # Penalty for invalid move
            return (np.array([self.N], dtype=np.int32), reward, True, False, {})

        # Perform the subtraction
        self.N -= prime_to_subtract

        if self.N == 0:
            # Current player wins by reducing N to zero
            self.done = True
            reward = 1
            return (np.array([self.N], dtype=np.int32), reward, True, False, {})
        else:
            # Check if the next player has any valid moves
            valid_moves_next_player = [
                idx for idx, prime in enumerate(self.allowed_primes) if prime <= self.N
            ]
            if not valid_moves_next_player:
                # Next player has no valid moves; current player wins
                self.done = True
                reward = 1
                return (np.array([self.N], dtype=np.int32), reward, True, False, {})
            else:
                # Game continues; switch to the other player
                self.current_player *= -1
                reward = 0
                return (np.array([self.N], dtype=np.int32), reward, False, False, {})

    def render(self):
        state_str = f"Current N: {self.N}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        state_str += "Available primes to subtract: "
        valid_moves = self.valid_moves()
        primes_available = [self.allowed_primes[i] for i in valid_moves]
        state_str += ", ".join(map(str, primes_available)) + "\n"
        return state_str

    def valid_moves(self):
        return [idx for idx, prime in enumerate(self.allowed_primes) if prime <= self.N]
