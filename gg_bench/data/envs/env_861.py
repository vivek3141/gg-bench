import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_number=100):
        super(CustomEnv, self).__init__()

        self.target_number = target_number

        # Define action space:
        # Actions 0-8: Append digits 1-9 (action + 1)
        # Action 9: Reverse number
        self.action_space = spaces.Discrete(10)

        # Observation space: [current_player_number, opponent_number, target_number]
        max_number = max(self.target_number * 10, 1e6)
        self.observation_space = spaces.Box(
            low=0, high=max_number, shape=(3,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_numbers = [0, 0]  # [player1_number, player2_number]
        self.current_player = 0  # 0 for player1, 1 for player2
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action (would exceed target number)
            reward = -10  # Penalty for invalid move
            terminated = True
            truncated = False
            self.done = True
            return self._get_observation(), reward, terminated, truncated, {}

        reward = -10  # Default reward for valid move
        terminated = False
        truncated = False

        current_number = self.player_numbers[self.current_player]

        # Process action
        if action in range(0, 9):  # Append digit (action + 1)
            digit_to_append = action + 1
            new_number = int(f"{current_number}{digit_to_append}")
        elif action == 9:  # Reverse number
            reversed_number_str = str(current_number)[::-1].lstrip("0") or "0"
            new_number = int(reversed_number_str)
        else:
            # Invalid action (should not be possible)
            reward = -10
            terminated = True
            truncated = False
            self.done = True
            return self._get_observation(), reward, terminated, truncated, {}

        self.player_numbers[self.current_player] = new_number

        # Check for win
        if new_number == self.target_number:
            reward = 1  # Reward for winning
            terminated = True
            self.done = True
        else:
            # Switch players
            self.current_player = 1 - self.current_player

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        s = f"Player {self.current_player + 1}'s Turn\n"
        s += f"Player 1 Current Number: {self.player_numbers[0]}\n"
        s += f"Player 2 Current Number: {self.player_numbers[1]}\n"
        s += f"Target Number: {self.target_number}\n"
        return s

    def valid_moves(self):
        valid_actions = []
        current_number = self.player_numbers[self.current_player]
        for action in range(10):
            # Simulate action
            if action in range(0, 9):
                # Append digit
                digit_to_append = action + 1
                new_number = int(f"{current_number}{digit_to_append}")
            elif action == 9:
                # Reverse number
                reversed_number_str = str(current_number)[::-1].lstrip("0") or "0"
                new_number = int(reversed_number_str)
            else:
                continue  # Should not reach here

            if new_number <= self.target_number:
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        return np.array(
            [
                self.player_numbers[self.current_player],
                self.player_numbers[1 - self.current_player],
                self.target_number,
            ],
            dtype=np.int32,
        )
