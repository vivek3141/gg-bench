import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the list of prime numbers
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        self.n_primes = len(self.primes)

        # Action space: indices from 0 to n_primes - 1
        self.action_space = spaces.Discrete(self.n_primes)

        # Observation space:
        # First n_primes elements: 0 = available, 1 = taken by player 1, 2 = taken by player 2
        # Next 2 elements: total sums of player 1 and player 2 (from 0 to 129)
        self.observation_space = spaces.Box(
            low=np.array([0] * self.n_primes + [0, 0]),
            high=np.array([2] * self.n_primes + [129, 129]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the game state
        self.prime_status = [
            0
        ] * self.n_primes  # 0 = available, 1 = player 1, 2 = player 2
        self.player_sums = [0, 0]  # Index 0: player 1's sum, Index 1: player 2's sum
        self.current_player = 0  # 0: player 1's turn, 1: player 2's turn
        self.done = False

        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            # If the game is over, return current state with zero reward
            return self._get_obs(), 0, self.done, False, {}

        # Check validity of action
        if action < 0 or action >= self.n_primes or self.prime_status[action] != 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move
        self.prime_status[action] = self.current_player + 1  # Set to 1 or 2
        # Update player's sum
        self.player_sums[self.current_player] += self.primes[action]

        # Check for win condition
        if self.player_sums[self.current_player] > 50:
            # Current player wins
            reward = 1
            self.done = True
        elif all(status != 0 for status in self.prime_status):
            # All primes have been selected
            # Determine winner based on total sums
            if self.player_sums[0] > self.player_sums[1]:
                winner = 0
            elif self.player_sums[1] > self.player_sums[0]:
                winner = 1
            else:
                # Equal sums, last player to have taken a turn loses
                winner = 1 - self.current_player
            if winner == self.current_player:
                reward = 1
            else:
                reward = -10
            self.done = True
        else:
            # Game continues
            reward = -10
            # Switch to the next player
            self.current_player = 1 - self.current_player

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        # Return the current observation
        observation = np.array(self.prime_status + self.player_sums, dtype=np.int32)
        return observation

    def render(self):
        # Return a string representation of the game state
        output = ""
        available_primes = [
            str(self.primes[i])
            for i in range(self.n_primes)
            if self.prime_status[i] == 0
        ]
        taken_primes = [
            f"{self.primes[i]}(P{self.prime_status[i]})"
            for i in range(self.n_primes)
            if self.prime_status[i] != 0
        ]
        output += "Available Primes: " + ", ".join(available_primes) + "\n"
        output += "Taken Primes: " + ", ".join(taken_primes) + "\n"
        output += f"Player 1 Total Sum: {self.player_sums[0]}\n"
        output += f"Player 2 Total Sum: {self.player_sums[1]}\n"
        if not self.done:
            output += f"Player {self.current_player + 1}'s turn.\n"
        else:
            output += "Game Over.\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices
        return [i for i, status in enumerate(self.prime_status) if status == 0]
