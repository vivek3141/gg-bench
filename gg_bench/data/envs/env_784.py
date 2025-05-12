import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sympy import primefactors


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # List of primes up to 50
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        self.prime_indices = {prime: idx for idx, prime in enumerate(self.primes)}

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.primes))
        self.observation_space = spaces.Box(low=1, high=50, shape=(3,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose a starting number between 15 and 50
        self.starting_number = np.random.randint(15, 51)
        self.player_numbers = [self.starting_number, self.starting_number]
        self.current_player = 0  # 0 or 1
        self.done = False

        observation = np.array(
            [
                self.player_numbers[self.current_player],
                self.player_numbers[1 - self.current_player],
                1 if self.current_player == 0 else -1,
            ],
            dtype=np.int32,
        )
        return observation, {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        selected_prime = self.primes[action]
        current_number = self.player_numbers[self.current_player]

        # Check if the move is valid
        valid_primes = primefactors(current_number)
        if selected_prime not in valid_primes:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Subtract the prime from the current player's number
        new_number = current_number - selected_prime
        if new_number <= 0:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        self.player_numbers[self.current_player] = new_number

        # Check for a win
        if new_number == 1:
            self.done = True
            return self.get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player

        # Prepare observation for the next player
        observation = self.get_observation()
        return observation, 0, False, False, {}

    def get_observation(self):
        observation = np.array(
            [
                self.player_numbers[self.current_player],
                self.player_numbers[1 - self.current_player],
                1 if self.current_player == 0 else -1,
            ],
            dtype=np.int32,
        )
        return observation

    def render(self):
        return (
            f"Player {self.current_player + 1}'s turn.\n"
            f"Your current number: {self.player_numbers[self.current_player]}\n"
            f"Opponent's number: {self.player_numbers[1 - self.current_player]}\n"
        )

    def valid_moves(self):
        current_number = self.player_numbers[self.current_player]
        valid_primes = primefactors(current_number)
        valid_actions = [self.prime_indices[prime] for prime in valid_primes]
        return valid_actions
