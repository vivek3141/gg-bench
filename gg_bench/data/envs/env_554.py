import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete space of numbers from 0 to 24 (representing numbers 1 to 25)
        self.action_space = spaces.Discrete(25)

        # Observation space: Box space containing the number pool status and players' scores
        # Number pool status: -1 (claimed by opponent), 0 (unclaimed), 1 (claimed by current player)
        # Players' scores: Any value from 0 to 325 (sum of numbers 1 to 25)
        # Observation shape: (27,) -> 25 for number pool, 2 for scores
        self.observation_space = spaces.Box(
            low=-1, high=325, shape=(27,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

        # Precompute prime numbers and prime factors for numbers 1 to 25
        self.primes = self._sieve_of_eratosthenes(25)
        self.prime_factors = self._compute_prime_factors(25)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool: 0 (unclaimed) for all numbers
        self.number_pool = np.zeros(25, dtype=np.int8)

        # Initialize scores for both players
        self.player_scores = {1: 0, -1: 0}

        # Set current player: 1 or -1
        self.current_player = 1

        # Game is not done
        self.done = False

        # Info dictionary can be extended to include additional information
        info = {}

        # Construct the observation
        observation = self._get_observation()
        return observation, info  # Return observation and info

    def step(self, action):
        # Check if the game is already done
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if the action is valid
        if action < 0 or action >= 25 or self.number_pool[action] != 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: update the number pool
        self.number_pool[action] = self.current_player

        # Get the number claimed (from 1 to 25)
        claimed_number = action + 1

        # Update scores based on the claimed number
        opponent = -self.current_player

        # Number 1: special case
        if claimed_number == 1:
            # No points awarded to either player
            pass
        # Prime number
        elif self.primes[claimed_number]:
            # Current player gains points equal to the prime number
            self.player_scores[self.current_player] += claimed_number
        # Composite number
        else:
            # Current player gains points equal to the composite number
            self.player_scores[self.current_player] += claimed_number

            # Opponent gains points equal to the sum of prime factors
            sum_prime_factors = sum(self.prime_factors[claimed_number])
            self.player_scores[opponent] += sum_prime_factors

        # Check if the game is over (all numbers have been claimed)
        if np.all(self.number_pool != 0):
            self.done = True
            # Determine the winner
            current_player_score = self.player_scores[self.current_player]
            opponent_score = self.player_scores[opponent]

            if current_player_score > opponent_score:
                # Current player wins
                reward = 1
            elif current_player_score < opponent_score:
                # Current player loses
                reward = -1
            else:
                # Tie: last player to have claimed a number wins
                reward = 1
        else:
            # Game continues
            reward = 0

        # Switch to the other player
        self.current_player *= -1

        # Return observation, reward, done, truncated, info
        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        # Create a visual representation of the game state
        output = "\nNumber Pool Status:\n"
        for i in range(1, 26):
            status = self.number_pool[i - 1]
            if status == 0:
                owner = "Unclaimed"
            elif status == self.current_player:
                owner = f"Player {self._player_str(self.current_player)}"
            else:
                owner = f"Player {self._player_str(-self.current_player)}"
            output += f"Number {i}: {owner}\n"

        # Display the players' scores
        output += "\nScores:\n"
        output += f"Player 1: {self.player_scores[1]}\n"
        output += f"Player 2: {self.player_scores[-1]}\n"

        # Indicate whose turn it is
        output += f"\nCurrent turn: Player {self._player_str(self.current_player)}\n"
        return output

    def valid_moves(self):
        # Return a list of valid moves (indices of unclaimed numbers)
        return [i for i in range(25) if self.number_pool[i] == 0]

    def _get_observation(self):
        # Construct the observation array
        observation = np.zeros(27, dtype=np.float32)
        # Number pool status
        observation[0:25] = self.number_pool
        # Current player's score
        observation[25] = self.player_scores[self.current_player]
        # Opponent's score
        opponent = -self.current_player
        observation[26] = self.player_scores[opponent]
        return observation

    def _player_str(self, player):
        return "1" if player == 1 else "2"

    def _sieve_of_eratosthenes(self, n):
        # Generate a list of primes up to n
        primes = [False, False] + [True] * (n - 1)
        for i in range(2, int(n**0.5) + 1):
            if primes[i]:
                for multiple in range(i * i, n + 1, i):
                    primes[multiple] = False
        return primes

    def _compute_prime_factors(self, n):
        # Compute prime factors for numbers from 2 to n
        prime_factors = {1: []}
        for num in range(2, n + 1):
            factors = []
            temp = num
            for i in range(2, num + 1):
                while temp % i == 0:
                    factors.append(i)
                    temp //= i
                if temp == 1:
                    break
            prime_factors[num] = factors
        return prime_factors
