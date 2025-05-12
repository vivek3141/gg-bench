import numpy as np
import gymnasium as gym
from gymnasium import spaces


def generate_primes_up_to(n):
    """Helper function to generate all prime numbers up to n."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i : n + 1 : i] = False
    primes = np.nonzero(sieve)[0]
    return primes.tolist()


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Generate all prime numbers up to N
        self.N = 2000  # You can adjust N as needed
        self.primes_list = generate_primes_up_to(self.N)
        self.primes_set = set(self.primes_list)
        self.num_primes = len(self.primes_list)

        # Map primes to indices for easy lookup
        self.prime_to_index = {prime: idx for idx, prime in enumerate(self.primes_list)}
        self.index_to_prime = {idx: prime for idx, prime in enumerate(self.primes_list)}

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_primes)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_primes + 1,), dtype=np.float32
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.used_primes = np.zeros(self.num_primes, dtype=np.float32)
        self.current_prime_index = -1  # -1 indicates no prime has been chosen yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Observation includes used_primes and current_prime_index normalized
        observation = np.append(
            self.used_primes, [self.current_prime_index / self.num_primes]
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over
            observation = np.append(
                self.used_primes, [self.current_prime_index / self.num_primes]
            )
            return observation, 0, True, False, {}
        if action < 0 or action >= self.num_primes:
            # Invalid action (out of bounds)
            reward = -10
            self.done = True
            observation = np.append(
                self.used_primes, [self.current_prime_index / self.num_primes]
            )
            return observation, reward, True, False, {}

        # Get the chosen prime number
        chosen_prime = self.index_to_prime[action]

        # Check if the prime has been used before
        if self.used_primes[action] == 1:
            reward = -10
            self.done = True
            observation = np.append(
                self.used_primes, [self.current_prime_index / self.num_primes]
            )
            return observation, reward, True, False, {}

        # First turn exception
        if self.current_prime_index == -1:
            if chosen_prime <= 2:
                # Invalid first move (must be greater than 2)
                reward = -10
                self.done = True
                observation = np.append(
                    self.used_primes, [self.current_prime_index / self.num_primes]
                )
                return observation, reward, True, False, {}
        else:
            current_prime = self.index_to_prime[self.current_prime_index]
            # Check if the chosen_prime satisfies the sequential relation
            if not (current_prime < chosen_prime < 2 * current_prime):
                reward = -10
                self.done = True
                observation = np.append(
                    self.used_primes, [self.current_prime_index / self.num_primes]
                )
                return observation, reward, True, False, {}

        # Valid move
        self.used_primes[action] = 1
        self.current_prime_index = action

        # Check if the next player has any valid moves
        self.current_player *= -1  # Switch player
        opponent_valid_moves = self.valid_moves()

        if not opponent_valid_moves:
            # Opponent cannot make a move, current player wins
            reward = 1
            self.done = True
            # Switch back to winning player for observation
            self.current_player *= -1
            observation = np.append(
                self.used_primes, [self.current_prime_index / self.num_primes]
            )
            return observation, reward, True, False, {}
        else:
            # Game continues
            reward = 0
            observation = np.append(
                self.used_primes, [self.current_prime_index / self.num_primes]
            )
            return observation, reward, False, False, {}

    def render(self):
        # Visual representation of the game state
        used_primes_indices = np.nonzero(self.used_primes)[0]
        used_primes = [self.index_to_prime[idx] for idx in used_primes_indices]

        if self.current_prime_index != -1:
            current_prime = self.index_to_prime[self.current_prime_index]
        else:
            current_prime = None

        render_str = f"Used Primes: {used_primes}\n"
        render_str += f"Current Prime: {current_prime}\n"
        render_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return render_str

    def valid_moves(self):
        valid_moves = []
        if self.current_prime_index == -1:
            # First turn, any prime greater than 2 not used
            for idx in range(self.num_primes):
                if self.used_primes[idx] == 0 and self.index_to_prime[idx] > 2:
                    valid_moves.append(idx)
        else:
            current_prime = self.index_to_prime[self.current_prime_index]
            lower_bound = current_prime
            upper_bound = 2 * current_prime
            for idx in range(self.num_primes):
                prime = self.index_to_prime[idx]
                if self.used_primes[idx] == 0 and lower_bound < prime < upper_bound:
                    valid_moves.append(idx)
        return valid_moves
