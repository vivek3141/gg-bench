import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_sum=30):
        super(CustomEnv, self).__init__()

        self.target_sum = target_sum

        # Define action and observation space
        # Actions are integers from 0 to 8, representing numbers 1 to 9
        self.action_space = spaces.Discrete(9)
        # Observation space contains the cumulative totals and current player
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]),
            high=np.array([self.target_sum, self.target_sum, 2]),
            shape=(3,),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_totals = np.array([0, 0], dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game is over

        # Map action 0-8 to number 1-9
        number_selected = action + 1

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, True, False, {}

        # Update the current player's cumulative total
        self.player_totals[self.current_player - 1] += number_selected

        # Check for win condition
        if self.player_totals[self.current_player - 1] == self.target_sum:
            self.done = True
            reward = 1  # Current player wins
            return self._get_obs(), reward, True, False, {}

        # Check for loss condition
        if self.player_totals[self.current_player - 1] > self.target_sum:
            self.done = True
            reward = -10  # Current player loses
            return self._get_obs(), reward, True, False, {}

        # No win or loss, game continues
        reward = 0

        # Switch to next player
        self.current_player = 1 if self.current_player == 2 else 2

        return self._get_obs(), reward, False, False, {}

    def render(self):
        return (
            f"Player 1 Total: {self.player_totals[0]}\n"
            f"Player 2 Total: {self.player_totals[1]}\n"
            f"Current Player: Player {self.current_player}\n"
        )

    def valid_moves(self):
        # Valid actions are those where the selected number does not cause cumulative total to exceed target sum
        current_total = self.player_totals[self.current_player - 1]
        max_add = self.target_sum - current_total
        # Actions are indices from 0 to 8 (representing numbers from 1 to 9)
        valid_actions = [action for action in range(9) if (action + 1) <= max_add]
        return valid_actions

    def _get_obs(self):
        # Observation is an array: [Player 1 Total, Player 2 Total, Current Player]
        return np.array(
            [self.player_totals[0], self.player_totals[1], self.current_player],
            dtype=np.int32,
        )
