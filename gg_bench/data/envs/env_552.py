import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 0 (select from start), 1 (select from end)
        self.action_space = spaces.Discrete(2)

        # Observation space: array of length 23
        # First 20 elements: sequence (numbers from 1 to 20, 0 if removed)
        # Next 2 elements: player totals
        # Last element: current player (0 or 1)
        self.observation_space = spaces.Box(low=0, high=50, shape=(23,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = list(range(1, 21))
        self.player_totals = [0, 0]  # [player 0 total, player 1 total]
        self.current_player = 0  # 0 for player 1, 1 for player 2
        self.done = False

        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            return self._get_obs(), -10, True, False, {}

        if len(self.sequence) == 0:
            # No more moves left
            self.done = True
            winner = self._determine_winner()
            if winner == self.current_player:
                reward = 1
            else:
                reward = -10
            return self._get_obs(), reward, True, False, {}

        # Select number from start or end
        if action == 0:
            number = self.sequence.pop(0)
        else:  # action == 1
            number = self.sequence.pop(-1)

        # Update current player's total
        self.player_totals[self.current_player] += number

        # Check for loss condition
        if self.player_totals[self.current_player] > 50:
            # Current player loses
            self.done = True
            # Since current player loses, reward is -10
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Check if sequence is exhausted
        if len(self.sequence) == 0:
            self.done = True
            winner = self._determine_winner()
            if winner == self.current_player:
                reward = 1
            else:
                reward = -10
            return self._get_obs(), reward, True, False, {}

        # Valid move
        reward = -10

        # Switch to next player
        self.current_player = 1 - self.current_player

        return self._get_obs(), reward, False, False, {}

    def render(self):
        sequence_str = "Current Sequence:\n" + str(self.sequence)
        totals_str = f"Player 1 Total: {self.player_totals[0]}\nPlayer 2 Total: {self.player_totals[1]}"
        current_player_str = f"Player {self.current_player +1}'s turn."
        return sequence_str + "\n" + totals_str + "\n" + current_player_str

    def valid_moves(self):
        if self.done or len(self.sequence) == 0:
            return []
        else:
            return [0, 1]

    def _get_obs(self):
        # Prepare the observation array
        obs_sequence = np.zeros(20, dtype=np.int32)
        for idx, num in enumerate(self.sequence):
            obs_sequence[idx] = num

        obs_totals = np.array(self.player_totals, dtype=np.int32)
        obs_current_player = np.array([self.current_player], dtype=np.int32)

        observation = np.concatenate([obs_sequence, obs_totals, obs_current_player])
        return observation

    def _determine_winner(self):
        if self.player_totals[0] > 50 and self.player_totals[1] > 50:
            # Both players exceeded 50, so both lose.
            # But according to the rules, the second player (player 1) wins in case of tie
            return 1  # Player 2 (index 1) wins
        elif self.player_totals[0] > 50:
            return 1  # Player 2 wins
        elif self.player_totals[1] > 50:
            return 0  # Player 1 wins
        else:
            # Both totals <= 50, compare totals
            if self.player_totals[0] < self.player_totals[1]:
                return 0  # Player 1 wins
            elif self.player_totals[1] < self.player_totals[0]:
                return 1  # Player 2 wins
            else:
                # Tie, second player wins
                return 1  # Player 2 wins
