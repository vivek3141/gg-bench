import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to numbers from 1 to 25 (indices 0 to 24)
        self.action_space = spaces.Discrete(25)

        # Observation consists of the claims array and the scores
        # Claims array: 0 (unclaimed), 1 (claimed by player 1), -1 (claimed by player 2)
        # Scores: current player's score, opponent's score
        # Observation shape is (27,)
        self.observation_space = spaces.Box(
            low=-1, high=100, shape=(27,), dtype=np.int16
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.claims = np.zeros(25, dtype=np.int8)  # Numbers 1 to 25 are unclaimed
        self.player_scores = [0, 0]  # Scores for player 1 and player 2
        self.current_player = 1  # Player 1 starts (represented by 1), Player 2 is -1
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, False, {"Error": "Game is over"}

        if action < 0 or action >= 25 or self.claims[action] != 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, self.done, False, {"Error": "Invalid move"}

        number = action + 1  # Numbers are from 1 to 25

        self.claims[action] = self.current_player

        # Indices for player scores
        current_player_index = 0 if self.current_player == 1 else 1
        opponent_player_index = 1 - current_player_index

        # Check if number is prime
        if self.is_prime(number):
            # Prime number, current player gains the number's value
            self.player_scores[current_player_index] += number
        else:
            # Composite number
            # Current player gains number's value
            self.player_scores[current_player_index] += number
            # Opponent gains sum of unique prime factors
            sum_pf = self.sum_of_prime_factors(number)
            self.player_scores[opponent_player_index] += sum_pf

        # Check victory conditions
        # If a player's score equals exactly 50, they win
        # If a player's score exceeds 50, they lose
        reward = 0
        self.done = False

        current_score = self.player_scores[current_player_index]
        opponent_score = self.player_scores[opponent_player_index]

        if current_score == 50:
            # Current player wins
            reward = 1
            self.done = True
        elif current_score > 50:
            # Current player loses
            reward = -1
            self.done = True
        elif opponent_score == 50:
            # Opponent wins
            reward = -1
            self.done = True
        elif opponent_score > 50:
            # Opponent loses, current player wins
            reward = 1
            self.done = True
        elif current_score > 50 and opponent_score > 50:
            # Both players exceed 50, active player loses
            reward = -1
            self.done = True

        if not self.done:
            self.current_player *= -1
            reward = -10  # As per the prompt

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        output = "--- Prime Claim ---\n\n"

        output += "Numbers Available: "
        for i in range(25):
            if self.claims[i] == 0:
                output += f"{i+1} "
        output += "\n\n"

        output += f"Player 1 Score: {self.player_scores[0]}\n"
        output += f"Player 2 Score: {self.player_scores[1]}\n\n"

        if not self.done:
            if self.current_player == 1:
                output += "Player 1, it's your turn.\n"
            else:
                output += "Player 2, it's your turn.\n"
        else:
            output += "Game over.\n"

        return output

    def valid_moves(self):
        return [i for i in range(25) if self.claims[i] == 0]

    def _get_obs(self):
        # Returns the observation as an array of 27 elements
        obs = np.zeros(27, dtype=np.int16)
        obs[:25] = self.claims
        # Scores from the perspective of the current player
        if self.current_player == 1:
            # Current player is Player 1
            obs[25] = self.player_scores[0]  # Current player's score
            obs[26] = self.player_scores[1]  # Opponent's score
        else:
            # Current player is Player 2
            obs[25] = self.player_scores[1]  # Current player's score
            obs[26] = self.player_scores[0]  # Opponent's score
        return obs

    def is_prime(self, n):
        if n <= 1:
            return False
        elif n <= 3:
            return True
        elif n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def sum_of_prime_factors(self, n):
        n_abs = abs(n)
        prime_factors = set()
        # Account for factor 2
        while n_abs % 2 == 0:
            prime_factors.add(2)
            n_abs //= 2
        # Account for factor 3
        while n_abs % 3 == 0:
            prime_factors.add(3)
            n_abs //= 3
        i = 5
        while i * i <= n_abs:
            while n_abs % i == 0:
                prime_factors.add(i)
                n_abs //= i
            while n_abs % (i + 2) == 0:
                prime_factors.add(i + 2)
                n_abs //= i + 2
            i += 6
        if n_abs > 1:
            prime_factors.add(n_abs)
        return sum(prime_factors)
