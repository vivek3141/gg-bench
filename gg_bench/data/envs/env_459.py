import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: select a number between 2 and 19 (inclusive)
        self.action_space = spaces.Discrete(
            18
        )  # Actions 0 to 17 correspond to numbers 2 to 19

        # Observation space:
        # - Player's collection: 3 numbers (0 if empty)
        # - Opponent's collection: 3 numbers (0 if empty)
        # - Target sum: integer between 20 and 40
        # - Current player indicator: 1 or -1
        low = np.array([0] * 6 + [20, -1], dtype=np.int32)
        high = np.array([19] * 6 + [40, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_sum = random.randint(20, 40)
        self.collections = {1: [], -1: []}  # Player 1: 1, Player 2: -1
        self.current_player = 1  # Start with Player 1
        self.done = False
        self.info = {}

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, self.info

        number = action + 2  # Map action to number between 2 and 19

        reward = 0
        terminated = False
        truncated = False

        # Check if current player already has three primes
        if len(self.collections[self.current_player]) >= 3:
            # Invalid move: cannot select more numbers
            reward = -10
            terminated = True
            self.done = True
            self.info["reason"] = (
                "Selected number when collection already has three primes."
            )
            return self._get_observation(), reward, terminated, truncated, self.info

        # Check if number is prime
        if not self._is_prime(number):
            # Non-prime number selected, discarded
            self._switch_player()
            observation = self._get_observation()
            reward = 0
            return observation, reward, False, False, {}
        else:
            # Prime number selected, add to collection
            self.collections[self.current_player].append(number)

            if len(self.collections[self.current_player]) == 3:
                total = sum(self.collections[self.current_player])
                if total == self.target_sum:
                    # Player wins
                    reward = 1
                    terminated = True
                    self.done = True
                    self.info["winner"] = self.current_player
                    self.info["reason"] = "Collected three primes summing to target."
                else:
                    # Player loses
                    reward = -10
                    terminated = True
                    self.done = True
                    self.info["winner"] = -self.current_player
                    self.info["reason"] = (
                        "Collected three primes not summing to target."
                    )
            else:
                # Turn passes to opponent
                self._switch_player()

        observation = self._get_observation()
        return observation, reward, terminated, truncated, self.info

    def render(self):
        current = f"Player {1 if self.current_player == 1 else 2}'s Turn\n"
        target = f"Target Sum: {self.target_sum}\n"
        p1_collection = f"Player 1's Collection: {self.collections[1]} Sum: {sum(self.collections[1]) if self.collections[1] else 0}\n"
        p2_collection = f"Player 2's Collection: {self.collections[-1]} Sum: {sum(self.collections[-1]) if self.collections[-1] else 0}\n"
        return current + target + p1_collection + p2_collection

    def valid_moves(self):
        if len(self.collections[self.current_player]) >= 3:
            return []
        else:
            return list(
                range(18)
            )  # All numbers from 2 to 19 are valid (indices 0 to 17)

    def _get_observation(self):
        player_collection = self.collections[self.current_player]
        opponent_collection = self.collections[-self.current_player]
        # Pad collections to length 3 with zeros
        player_padded = player_collection + [0] * (3 - len(player_collection))
        opponent_padded = opponent_collection + [0] * (3 - len(opponent_collection))
        observation = np.array(
            player_padded + opponent_padded + [self.target_sum, self.current_player],
            dtype=np.int32,
        )
        return observation

    def _switch_player(self):
        self.current_player *= -1

    def _is_prime(self, num):
        if num < 2:
            return False
        for n in range(2, int(num**0.5) + 1):
            if num % n == 0:
                return False
        return True
