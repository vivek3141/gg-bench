import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_N=100):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete space of primes up to starting_N
        self.starting_N = starting_N
        self.primes = self.get_primes_up_to_N(self.starting_N)
        self.num_primes = len(self.primes)
        self.action_space = spaces.Discrete(self.num_primes)

        # Observation space: current N and current player
        # N ranges from 1 to starting_N, current player is -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([1, -1]),
            high=np.array([self.starting_N, 1]),
            shape=(2,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.observation(), {}

    def step(self, action):
        if self.done:
            return self.observation(), 0, True, False, {}

        # Check if there are valid moves
        if len(self.valid_moves()) == 0:
            # Current player cannot move, loses
            self.done = True
            reward = -1  # Losing reward
            return self.observation(), reward, True, False, {}

        # Map action to prime
        p = self.primes[action]

        # Check if p is valid prime factor of N
        if self.N % p != 0:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.observation(), reward, True, False, {}

        # Valid move
        self.N = self.N // p

        # Switch player
        self.current_player *= -1

        # Check if next player has any valid moves
        if len(self.valid_moves()) == 0:
            # Next player cannot move, current player wins
            # Switch back to winning player for correct observation
            self.current_player *= -1
            self.done = True
            reward = +1  # Winning reward
            return self.observation(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            return self.observation(), reward, False, False, {}

    def observation(self):
        return np.array([self.N, self.current_player], dtype=np.int32)

    def render(self):
        current_player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current player: {current_player_str}, N = {self.N}"

    def valid_moves(self):
        valid_actions = []
        for idx, p in enumerate(self.primes):
            if self.N % p == 0:
                valid_actions.append(idx)
        return valid_actions

    @staticmethod
    def get_primes_up_to_N(N):
        sieve = [True] * (N + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(N**0.5) + 1):
            if sieve[i]:
                sieve[i * i : N + 1 : i] = [False] * len(range(i * i, N + 1, i))
        return [i for i, is_prime in enumerate(sieve) if is_prime]
