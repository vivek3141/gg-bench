import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int32)

        # Set up the game parameters
        self.target_sum = 17  # This can be changed as per the game's requirement
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = np.zeros(
            9, dtype=np.int32
        )  # 0: unclaimed, 1: player 1, -1: player -1
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.player_collections = {1: [], -1: []}
        return self.observation, {}  # Return observation and info

    def step(self, action):
        number = action + 1  # Map action to number 1-9
        if self.observation[action] != 0 or self.done:
            # Invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        # Valid move
        self.observation[action] = self.current_player
        self.player_collections[self.current_player].append(number)
        player_total = sum(self.player_collections[self.current_player])

        # Check for win
        if self.is_prime(player_total) and player_total >= self.target_sum:
            # Current player wins
            self.done = True
            return self.observation, 1, True, False, {}

        # Check if all numbers are picked
        if np.all(self.observation != 0):
            # Game over, determine winner
            opponent = -self.current_player
            opponent_total = sum(self.player_collections[opponent])
            player_high_prime = self.get_highest_prime_leq(player_total)
            opponent_high_prime = self.get_highest_prime_leq(opponent_total)
            if player_high_prime > opponent_high_prime:
                # Current player wins
                self.done = True
                return self.observation, 1, True, False, {}
            elif opponent_high_prime > player_high_prime:
                # Current player loses
                self.done = True
                return self.observation, -1, True, False, {}
            else:
                # Same highest prime totals, last player to move loses
                self.done = True
                return self.observation, -1, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return self.observation, 0, False, False, {}

    def render(self):
        # Create a visual representation of the environment state
        state_str = "Available Numbers: "
        for i, val in enumerate(self.observation):
            if val == 0:
                state_str += f"{i+1} "
        state_str += "\n"
        state_str += f"Player 1 Collection: {self.player_collections[1]}\n"
        state_str += f"Player 1 Total Sum: {sum(self.player_collections[1])}\n"
        state_str += f"Player -1 Collection: {self.player_collections[-1]}\n"
        state_str += f"Player -1 Total Sum: {sum(self.player_collections[-1])}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return state_str

    def valid_moves(self):
        return [i for i in range(9) if self.observation[i] == 0]

    def is_prime(self, n):
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

    def get_highest_prime_leq(self, n):
        for num in range(n, 1, -1):
            if self.is_prime(num):
                return num
        return 0  # No primes less than or equal to n

    def highest_prime_subset_sum(self, numbers):
        max_prime = 0
        for r in range(1, len(numbers) + 1):
            for subset in combinations(numbers, r):
                s = sum(subset)
                if self.is_prime(s) and s > max_prime:
                    max_prime = s
        return max_prime
