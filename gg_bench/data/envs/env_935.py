import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # List of prime numbers between 2 and 50
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        self.num_primes = len(self.primes)

        # Map primes to indices
        self.prime_to_index = {prime: idx for idx, prime in enumerate(self.primes)}
        self.index_to_prime = {idx: prime for idx, prime in enumerate(self.primes)}

        # Action space: indices of primes
        self.action_space = spaces.Discrete(self.num_primes)

        # Observation space:
        # - First 'num_primes' elements: availability of primes (1 available, 0 not)
        # - Next 10 elements: digits to avoid (digits 0-9)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_primes + 10,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All primes are initially available
        self.available_primes = np.ones(self.num_primes, dtype=np.int32)
        # No digits to avoid at the beginning
        self.digits_to_avoid = np.zeros(10, dtype=np.int32)
        # Initialize last opponent's pick to None
        self.last_opponent_pick = None
        # Current player (1 or -1)
        self.current_player = 1
        # Game is not done
        self.done = False

        # Observation is the concatenation of available primes and digits to avoid
        observation = np.concatenate([self.available_primes, self.digits_to_avoid])

        return observation, {}

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if action is valid
        reward = 0
        terminated = False
        truncated = False

        prime = self.index_to_prime[action]
        is_valid_move = self._is_valid_move(prime)

        if not is_valid_move:
            # Invalid move
            reward = -10
            terminated = True
            self.done = True
            return self._get_obs(), reward, terminated, truncated, {}

        # Valid move
        # Update the state
        self.available_primes[action] = 0

        # Update the digits to avoid for the opponent's next turn
        self.digits_to_avoid = self._get_digits(prime)

        # Check if the game has ended
        if np.sum(self.available_primes) == 0:
            # No more primes available; current player wins
            reward = 1
            terminated = True
            self.done = True
            return self._get_obs(), reward, terminated, truncated, {}

        # Check if opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot make a valid move; current player wins
            reward = 1
            terminated = True
            self.done = True
            return self._get_obs(), reward, terminated, truncated, {}

        # Switch to opponent's turn
        self.current_player *= -1
        self.last_opponent_pick = prime

        # Return the observation
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # Display the current game state
        available_primes_list = [
            self.primes[i] for i in range(self.num_primes) if self.available_primes[i]
        ]
        digits_to_avoid_list = [str(i) for i in range(10) if self.digits_to_avoid[i]]
        state_str = "Available Primes: " + str(available_primes_list) + "\n"
        state_str += "Digits to Avoid: " + str(digits_to_avoid_list) + "\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid moves (action indices)
        valid_actions = []
        for idx in range(self.num_primes):
            if self.available_primes[idx]:
                prime = self.primes[idx]
                if self._is_valid_move(prime):
                    valid_actions.append(idx)
        return valid_actions

    def _is_valid_move(self, prime):
        # Check if the prime is available
        idx = self.prime_to_index[prime]
        if not self.available_primes[idx]:
            return False
        # Check if prime shares digits with opponent's last pick
        digits_in_prime = self._get_digits(prime)
        if np.any(np.logical_and(digits_in_prime, self.digits_to_avoid)):
            return False
        return True

    def _get_digits(self, number):
        # Returns a binary vector of digits present in the number
        digits = np.zeros(10, dtype=np.int32)
        for digit_char in str(number):
            digits[int(digit_char)] = 1
        return digits

    def _get_obs(self):
        # Returns the current observation
        observation = np.concatenate([self.available_primes, self.digits_to_avoid])
        return observation
