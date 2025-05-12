import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting numbers 1-9 (indices 0-8)
        self.action_space = spaces.Discrete(9)

        # Observation space represents the shared pool and selections:
        # 0: number is in the shared pool
        # 1: number selected by current player
        # -1: number selected by opponent
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Shared pool: 1 if number is available, 0 if taken
        self.shared_pool = [1] * 9
        # Sequences and sums for both players
        self.player_sequences = {1: [], -1: []}
        self.player_sums = {1: 0, -1: 0}
        # Current player: 1 (first player), -1 (second player)
        self.current_player = 1
        # Game over flag
        self.done = False

        # Return initial observation and info
        return np.zeros(9, dtype=np.int8), {}

    def step(self, action):
        action = int(action)
        number = action + 1  # Map action index to number (1-9)

        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if the selected number is available
        if self.shared_pool[action] == 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: update shared pool and player's sequence and sum
        self.shared_pool[action] = 0
        self.player_sequences[self.current_player].append(number)
        self.player_sums[self.current_player] += number

        # Generate observation
        observation = self._get_observation()

        # Initialize reward and termination flag
        reward = 0
        terminated = False

        # Check for winning conditions
        if self.player_sums[self.current_player] == 15:
            # Exact Sum Victory
            reward = 1
            terminated = True
            self.done = True
        elif all(s == 0 for s in self.shared_pool):
            # All numbers exhausted
            other_player = -self.current_player
            current_sum = self.player_sums[self.current_player]
            other_sum = self.player_sums[other_player]

            if current_sum <= 15 and (current_sum > other_sum or other_sum > 15):
                reward = 1
            elif current_sum == other_sum:
                # Tie-Breaker: fewer numbers in sequence wins
                current_len = len(self.player_sequences[self.current_player])
                other_len = len(self.player_sequences[other_player])
                if current_len < other_len:
                    reward = 1
                elif current_len == other_len:
                    # Second player wins the tie if still tied
                    if self.current_player == -1:
                        reward = 1
                    else:
                        reward = -1
                else:
                    reward = -1
            else:
                reward = -1
            terminated = True
            self.done = True
        else:
            # Switch to the other player
            self.current_player *= -1

        # Return observation, reward, termination flag, and info
        return observation, reward, terminated, False, {}

    def render(self):
        # Generate a visual representation of the game state
        pool_numbers = [i + 1 for i in range(9) if self.shared_pool[i] == 1]
        current_sequence = self.player_sequences[self.current_player]
        current_sum = self.player_sums[self.current_player]
        other_player = -self.current_player
        opponent_sequence = self.player_sequences[other_player]
        opponent_sum = self.player_sums[other_player]

        output = f"Shared Pool: {pool_numbers}\n"
        output += f"Player {self.current_player} Sequence: {current_sequence} | Sum: {current_sum}\n"
        output += f"Player {other_player} Sequence: {opponent_sequence} | Sum: {opponent_sum}\n"
        return output

    def valid_moves(self):
        # Return a list of valid moves (available numbers)
        return [i for i in range(9) if self.shared_pool[i] == 1]

    def _get_observation(self):
        # Create an observation array based on the shared pool and selections
        obs = np.zeros(9, dtype=np.int8)
        for i in range(9):
            if self.shared_pool[i] == 0:
                # Number is taken
                number_taken = i + 1
                if number_taken in self.player_sequences[self.current_player]:
                    obs[i] = 1
                else:
                    obs[i] = -1
            else:
                obs[i] = 0
        return obs
