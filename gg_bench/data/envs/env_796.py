import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 1 to 6, represented as actions 0 to 5
        self.action_space = spaces.Discrete(6)

        # Observation space: [current_player_sum, opponent_sum, opponent_last_choice]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]), high=np.array([31, 31, 6]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_sums = [0, 0]
        self.opponent_last_choice = None  # Initially None

        self.current_player = 0  # 0 or 1

        self.done = False

        # Return the initial observation
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if not valid_actions:
            # No valid moves, player loses
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        if action not in valid_actions:
            self.done = True
            reward = -10  # Invalid move
            return self._get_observation(), reward, True, False, {}

        number_chosen = action + 1  # Convert action index to number

        # Update current player's sum
        self.player_sums[self.current_player] += number_chosen

        # Check for overstepping
        if self.player_sums[self.current_player] > 31:
            self.done = True
            reward = -10  # Overstepping, lose
            return self._get_observation(), reward, True, False, {}

        # Check for winning
        if self.player_sums[self.current_player] == 31:
            self.done = True
            reward = 1  # Winning
            return self._get_observation(), reward, True, False, {}

        # Move is valid and game continues
        reward = 0

        # Update opponent's last choice
        self.opponent_last_choice = number_chosen

        # Switch to opponent
        self.current_player = 1 - self.current_player

        return self._get_observation(), reward, False, False, {}

    def _get_observation(self):
        observation = np.array(
            [
                self.player_sums[self.current_player],
                self.player_sums[1 - self.current_player],
                (
                    self.opponent_last_choice
                    if self.opponent_last_choice is not None
                    else 0
                ),
            ],
            dtype=np.int32,
        )
        return observation

    def render(self):
        return f"Player {self.current_player + 1}'s turn. Your total sum: {self.player_sums[self.current_player]}. Opponent's sum: {self.player_sums[1 - self.current_player]}. Opponent's last choice: {self.opponent_last_choice}."

    def valid_moves(self):
        valid_moves = []
        opponent_last = self.opponent_last_choice
        for i in range(1, 7):
            if opponent_last is None or i != opponent_last:
                if self.player_sums[self.current_player] + i <= 31:
                    valid_moves.append(i - 1)  # action indices are 0-based
        return valid_moves
