import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=20):
        super(CustomEnv, self).__init__()
        self.starting_number = starting_number
        self.current_number = starting_number
        self.current_player = 1

        # Define action and observation spaces
        # Actions: Numbers from 1 to starting_number inclusive
        self.action_space = spaces.Discrete(
            self.starting_number + 1
        )  # Actions from 0 to starting_number inclusive
        # Observations: [Current Number, Current Player]
        self.observation_space = spaces.Box(
            low=np.array([0, 1], dtype=np.int32),
            high=np.array([self.starting_number, 2], dtype=np.int32),
            dtype=np.int32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1
        return self._get_obs(), {}

    def step(self, action):
        # Check if action is valid (number to subtract)
        if action < 1 or action > self.current_number:
            # Invalid action
            return self._get_obs(), -10, True, False, {}

        # Check if action is valid according to game rules
        if self.is_prime(self.current_number) or self.current_number == 1:
            # Must subtract 1
            if action != 1:
                # Invalid action
                return self._get_obs(), -10, True, False, {}
        else:
            # Action must be a proper divisor excluding 1 and the number itself
            proper_divisors = self.get_proper_divisors(self.current_number)
            if action not in proper_divisors:
                # Invalid action
                return self._get_obs(), -10, True, False, {}

        # Valid action: Subtract and update the current number
        self.current_number -= action

        # Check for win condition
        if self.current_number == 0:
            # Current player wins
            return self._get_obs(), 1, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_obs(), 0, False, False, {}

    def render(self):
        render_str = "--- Divisor Duel ---\n\n"
        render_str += f"Player {self.current_player}'s Turn\n"
        render_str += f"Current Number: {self.current_number}\n"
        if self.is_prime(self.current_number):
            render_str += f"{self.current_number} is a prime number.\n"
            render_str += "You must subtract 1.\n"
        elif self.current_number == 1:
            render_str += "No proper divisors available.\n"
            render_str += "You must subtract 1.\n"
        else:
            proper_divisors = self.get_proper_divisors(self.current_number)
            render_str += f"Proper Divisors: {proper_divisors}\n"
        return render_str

    def valid_moves(self):
        valid_actions = []

        if self.current_number == 0:
            return valid_actions  # No moves available

        if self.is_prime(self.current_number) or self.current_number == 1:
            valid_actions.append(1)
        else:
            proper_divisors = self.get_proper_divisors(self.current_number)
            valid_actions = proper_divisors[:]

        return valid_actions

    def _get_obs(self):
        return np.array([self.current_number, self.current_player], dtype=np.int32)

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

    def get_proper_divisors(self, n):
        divisors = []
        for i in range(2, n // 2 + 1):
            if n % i == 0:
                divisors.append(i)
        return divisors
