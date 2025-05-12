import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def unique_prime_factors(n):
    factors = set()
    # Handle 2 separately
    while n % 2 == 0:
        factors.add(2)
        n //= 2
    # Check for odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.add(i)
            n //= i
        i += 2
    # If n is a prime number greater than 2
    if n > 2:
        factors.add(n)
    return factors


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(19) corresponding to numbers 2 to 20 inclusive
        self.action_space = spaces.Discrete(19)

        # Observation space: [current_player_life_points, opponent_life_points]
        # We set high to 1000 assuming life points won't exceed this in usual gameplay
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(2,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.life_points = [50, 50]  # [Player 0 life points, Player 1 life points]
        self.current_player = 0  # 0 or 1
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Observation is from the perspective of the current player
        return np.array(
            [
                self.life_points[self.current_player],
                self.life_points[1 - self.current_player],
            ],
            dtype=np.int32,
        )

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        chosen_number = (
            action + 2
        )  # Map action index to number between 2 and 20 inclusive

        # Process the action
        if is_prime(chosen_number):
            # Attack: Deal damage equal to the prime number to opponent
            damage = chosen_number
            self.life_points[1 - self.current_player] -= damage
        else:
            # Heal: Heal self by the number of unique prime factors
            factors = unique_prime_factors(chosen_number)
            heal_amount = len(factors)
            self.life_points[self.current_player] += heal_amount

        # Check if the opponent is defeated
        if self.life_points[1 - self.current_player] <= 0:
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player

        return self._get_obs(), 0, False, False, {}

    def render(self):
        return (
            f"Player {self.current_player}'s turn.\n"
            f"Your life points: {self.life_points[self.current_player]}.\n"
            f"Opponent's life points: {self.life_points[1 - self.current_player]}."
        )

    def valid_moves(self):
        # All actions between 0 and 18 are valid since numbers can be reused
        return list(range(19))
