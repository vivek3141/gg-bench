import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(10), corresponding to attack numbers 1 to 10
        self.action_space = spaces.Discrete(10)

        # Observation space: LP1, LP2, player1_used_numbers (10), player2_used_numbers (10)
        # LPs range from -1000 to 100, used_numbers are 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([-1000, -1000] + [0] * 20, dtype=np.int32),
            high=np.array([100, 100] + [1] * 20, dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_lp = 100
        self.player2_lp = 100
        self.player1_used_numbers = np.zeros(10, dtype=np.int32)
        self.player2_used_numbers = np.zeros(10, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        observation = np.concatenate(
            [
                np.array([self.player1_lp, self.player2_lp], dtype=np.int32),
                self.player1_used_numbers,
                self.player2_used_numbers,
            ]
        )

        return observation

    def step(self, action):
        if self.done:
            # If the game is over, no further action can be taken
            return self._get_observation(), 0, True, False, {}

        action_number = action + 1  # Action 0 corresponds to number 1

        # Determine which player's turn it is
        if self.current_player == 1:
            # Check if number has already been used by player 1
            if self.player1_used_numbers[action] == 1:
                # Invalid move
                reward = -10
                terminated = False
                truncated = False
                self.current_player = 2  # Switch to player 2
                observation = self._get_observation()
                return observation, reward, terminated, truncated, {}
            else:
                # Valid move
                self.player1_used_numbers[action] = 1  # Mark number as used
                self.player2_lp -= action_number  # Subtract from opponent's LP

                # Check for win condition
                if self.player2_lp <= 0:
                    reward = 1
                    terminated = True
                    truncated = False
                    self.done = True
                else:
                    reward = 0
                    terminated = False
                    truncated = False
                    self.current_player = 2  # Switch to player 2

                observation = self._get_observation()
                return observation, reward, terminated, truncated, {}
        elif self.current_player == 2:
            # Check if number has already been used by player 2
            if self.player2_used_numbers[action] == 1:
                # Invalid move
                reward = -10
                terminated = False
                truncated = False
                self.current_player = 1  # Switch to player 1
                observation = self._get_observation()
                return observation, reward, terminated, truncated, {}
            else:
                # Valid move
                self.player2_used_numbers[action] = 1  # Mark number as used
                self.player1_lp -= action_number  # Subtract from opponent's LP

                # Check for win condition
                if self.player1_lp <= 0:
                    reward = 1
                    terminated = True
                    truncated = False
                    self.done = True
                else:
                    reward = 0
                    terminated = False
                    truncated = False
                    self.current_player = 1  # Switch to player 1

                observation = self._get_observation()
                return observation, reward, terminated, truncated, {}

    def render(self):
        result = "--- Number Wars Game State ---\n"
        result += f"Player 1 LP: {self.player1_lp}\n"
        result += f"Player 2 LP: {self.player2_lp}\n"
        result += f"Player {self.current_player}'s turn.\n"
        used_numbers_p1 = np.where(self.player1_used_numbers == 1)[0] + 1
        used_numbers_p2 = np.where(self.player2_used_numbers == 1)[0] + 1
        result += f"Player 1 Used Numbers: {used_numbers_p1.tolist()}\n"
        result += f"Player 2 Used Numbers: {used_numbers_p2.tolist()}\n"
        return result

    def valid_moves(self):
        if self.current_player == 1:
            return [i for i in range(10) if self.player1_used_numbers[i] == 0]
        elif self.current_player == 2:
            return [i for i in range(10) if self.player2_used_numbers[i] == 0]
