import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting numbers from 2 to 20 inclusive
        self.action_space = spaces.Discrete(19)
        # Observation space consists of:
        # - Player's HP
        # - Opponent's HP
        # - Availability of numbers from 2 to 20 (1 if available, 0 if used)
        self.observation_space = spaces.Box(low=0, high=20, shape=(21,), dtype=np.int32)

        # Map action indices to numbers from 2 to 20
        self.action_to_number = {i: num for i, num in enumerate(range(2, 21))}
        self.number_to_action = {num: i for i, num in self.action_to_number.items()}

        # Precompute prime status and prime factors for numbers 2 to 20
        self.prime_status = {num: self.is_prime(num) for num in range(2, 21)}
        self.prime_factors_cache = {
            num: self.get_prime_factors(num) for num in range(2, 21)
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [20, 20]  # [Player 0 HP, Player 1 HP]
        self.numbers_available = np.ones(
            19, dtype=np.int32
        )  # 1 if number is available, 0 if used
        self.current_player = 0  # 0 or 1
        self.done = False
        self.reward = 0
        return self._get_observation(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Check if action is valid
        if action < 0 or action >= 19 or self.numbers_available[action] == 0:
            # Invalid move
            self.done = True
            self.reward = -10
            return self._get_observation(), self.reward, True, False, {}

        # Valid move
        number = self.action_to_number[action]
        self.numbers_available[action] = 0  # Mark number as used

        # Calculate damage
        damage = self.calculate_damage(number)

        # Apply damage to opponent
        opponent = 1 - self.current_player
        self.player_hp[opponent] -= damage
        if self.player_hp[opponent] < 0:
            self.player_hp[opponent] = 0  # HP cannot go below zero

        # Check for game over
        if self.player_hp[opponent] <= 0:
            # Current player wins
            self.done = True
            self.reward = 1
            return self._get_observation(), self.reward, True, False, {}
        else:
            # Continue game
            self.reward = 0
            # Switch to next player
            self.current_player = opponent
            return self._get_observation(), self.reward, False, False, {}

    def render(self):
        lines = []
        lines.append(f"Player {self.current_player + 1}'s Turn")
        lines.append(f"Player 1 HP: {self.player_hp[0]}")
        lines.append(f"Player 2 HP: {self.player_hp[1]}")
        available_numbers = [
            self.action_to_number[i]
            for i in range(19)
            if self.numbers_available[i] == 1
        ]
        lines.append(f"Available Numbers: {available_numbers}")
        return "\n".join(lines)

    def valid_moves(self):
        return [i for i in range(19) if self.numbers_available[i] == 1]

    def _get_observation(self):
        # Observation consists of:
        # [Current player's HP, Opponent's HP, Number availability (19 numbers)]
        opponent = 1 - self.current_player
        observation = np.concatenate(
            (
                np.array(
                    [self.player_hp[self.current_player], self.player_hp[opponent]],
                    dtype=np.int32,
                ),
                self.numbers_available.copy(),
            )
        )
        return observation

    @staticmethod
    def is_prime(n):
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

    def get_prime_factors(self, n):
        factors = []
        # Handle 2 separately
        while n % 2 == 0:
            factors.append(2)
            n = n // 2
        # Handle odd factors
        p = 3
        while p * p <= n:
            while n % p == 0:
                factors.append(p)
                n = n // p
            p += 2
        if n > 1:
            factors.append(n)
        return factors

    def calculate_damage(self, number):
        if self.prime_status[number]:
            # Prime number
            damage = number
        else:
            # Composite number
            factors = self.prime_factors_cache[number]
            damage = sum(factors)
        return damage
