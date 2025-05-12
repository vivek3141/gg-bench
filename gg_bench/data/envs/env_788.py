import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is numbers from 1 to 10, represented as indices 0 to 9
        self.action_space = spaces.Discrete(10)

        # Observation space consists of:
        # - Numbers available in the pool (positions 0-9): 0 (not available) or 1 (available)
        # - Current player's total sum (position 10): 0 to 25
        # - Opponent's total sum (position 11): 0 to 25
        self.observation_space = spaces.Box(low=0, high=25, shape=(12,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool: numbers 1 to 10 are available (1), not selected (0)
        self.number_pool = np.ones(10, dtype=np.int8)
        # Player totals
        self.player_totals = {1: 0, 2: 0}
        # Start with Player 1
        self.current_player = 1
        self.done = False
        # Build the initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action index to number (1-10)
        number_selected = action + 1

        # Check if the selected number is available in the pool
        if self.number_pool[action] == 0:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Update the current player's total sum
        self.player_totals[self.current_player] += number_selected
        # Remove the selected number from the pool
        self.number_pool[action] = 0

        # Check for winning condition
        if self.player_totals[self.current_player] == 25:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Check if current player's total exceeds 25
        if self.player_totals[self.current_player] > 25:
            # Current player loses
            self.done = True
            reward = -1
            return self._get_observation(), reward, True, False, {}

        # Check if all numbers have been selected
        if np.sum(self.number_pool) == 0:
            # Game ends, compare totals
            self.done = True
            player_total = self.player_totals[self.current_player]
            opponent_total = self.player_totals[3 - self.current_player]
            if player_total > opponent_total:
                reward = 1
            else:
                reward = -1
            return self._get_observation(), reward, True, False, {}

        # Game continues, switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Generate visual representation
        available_numbers = [str(i + 1) for i in range(10) if self.number_pool[i] == 1]
        render_str = "Numbers remaining: " + ", ".join(available_numbers) + "\n"
        render_str += f"Player {self.current_player} total: {self.player_totals[self.current_player]}\n"
        render_str += f"Player {3 - self.current_player} total: {self.player_totals[3 - self.current_player]}\n"
        return render_str

    def valid_moves(self):
        # Return indices of available numbers
        return [i for i in range(10) if self.number_pool[i] == 1]

    def _get_observation(self):
        # Build observation array
        observation = np.zeros(12, dtype=np.int8)
        # Numbers available in the pool
        observation[0:10] = self.number_pool
        # Current player's total sum
        observation[10] = self.player_totals[self.current_player]
        # Opponent's total sum
        observation[11] = self.player_totals[3 - self.current_player]
        return observation
