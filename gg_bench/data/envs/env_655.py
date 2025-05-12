import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # List of prime numbers
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        # Define the action space: Discrete space with 10 actions (indices of primes)
        self.action_space = spaces.Discrete(len(self.primes))
        # Define the observation space
        # Observation includes:
        # - Current player's life total (float)
        # - Opponent's life total (float)
        # - Available primes (binary vector of length 10)
        low = np.array([-100.0, -100.0] + [0] * len(self.primes), dtype=np.float32)
        high = np.array([50.0, 50.0] + [1] * len(self.primes), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Players' life totals: [Player 0's life, Player 1's life]
        self.life_totals = [50, 50]
        # Primes availability: 1 if available, 0 if used
        self.primes_available = [1] * len(self.primes)
        # Current player: 0 or 1
        self.current_player = 0
        # Game over flag
        self.done = False
        return self._get_obs(), {}  # Return observation and empty info dict

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_obs(), 0, True, False, {}  # No reward if the game is over

        # Check if the action is valid
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, True, False, {}

        # Perform the action
        # Remove the selected prime from the available pool
        self.primes_available[action] = 0
        # Subtract the prime number from the opponent's life total
        opponent = 1 - self.current_player
        self.life_totals[opponent] -= self.primes[action]

        # Check for victory conditions
        if self.life_totals[opponent] <= 0:
            # Opponent's life total is zero or below; current player wins
            self.done = True
            reward = 1  # Positive reward for winning
            return self._get_obs(), reward, True, False, {}
        elif sum(self.primes_available) == 0:
            # All primes have been used; check life totals
            if self.life_totals[self.current_player] > self.life_totals[opponent]:
                # Current player has higher life total; wins the game
                self.done = True
                reward = 1  # Positive reward for winning
                return self._get_obs(), reward, True, False, {}
            else:
                # Current player loses
                self.done = True
                reward = -10  # Penalty for losing
                return self._get_obs(), reward, True, False, {}
        else:
            # Game continues
            reward = -10  # Penalty for a valid move to encourage quick victory
            self.current_player = opponent  # Switch turns
            return self._get_obs(), reward, False, False, {}

    def render(self):
        # Return a string representation of the game state
        output = f"Player {self.current_player + 1}'s Turn\n"
        output += f"Player 1 Life Total: {self.life_totals[0]}\n"
        output += f"Player 2 Life Total: {self.life_totals[1]}\n"
        available_primes = [
            str(self.primes[i])
            for i in range(len(self.primes))
            if self.primes_available[i]
        ]
        output += "Available Primes: " + ", ".join(available_primes) + "\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices (available primes)
        return [i for i in range(len(self.primes)) if self.primes_available[i] == 1]

    def _get_obs(self):
        # Return the current observation array
        obs = np.array(
            [
                self.life_totals[self.current_player],
                self.life_totals[1 - self.current_player],
            ]
            + self.primes_available,
            dtype=np.float32,
        )
        return obs
