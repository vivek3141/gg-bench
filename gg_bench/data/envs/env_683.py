import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are numbers from 1 to 9 (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space consists of:
        # - Own cumulative total (0 to 50)
        # - Opponent's cumulative total (0 to 50)
        # - Opponent's last selected number (0 to 9)
        low = np.array([0, 0, 0], dtype=np.int32)
        high = np.array([50, 50, 9], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_total = 0
        self.player2_total = 0
        self.opponent_last_number = None  # No last number at the start
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            observation = self._get_observation()
            info = {}
            return (
                observation,
                0,
                self.done,
                False,
                info,
            )  # reward, terminated, truncated, info

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            reward = -10  # Invalid move penalty
            observation = self._get_observation()
            info = {}
            return observation, reward, self.done, False, info

        # Action is valid
        number_selected = action + 1  # Map action index to number 1-9
        if self.current_player == 1:
            self.player1_total += number_selected
            player_total = self.player1_total
        else:
            self.player2_total += number_selected
            player_total = self.player2_total

        # Check for exceeding 50
        if player_total > 50:
            self.done = True
            reward = -10  # Player loses by exceeding 50
            observation = self._get_observation()
            info = {}
            return observation, reward, self.done, False, info
        elif player_total == 50:
            self.done = True
            reward = 1  # Player wins by reaching exactly 50
            observation = self._get_observation()
            info = {}
            return observation, reward, self.done, False, info
        else:
            # Game continues
            # Update opponent's last number
            self.opponent_last_number = number_selected
            # Swap current player
            self.current_player = 1 if self.current_player == 2 else 2
            # Next player receives observation
            observation = self._get_observation()
            reward = 0  # No immediate reward
            info = {}
            return observation, reward, self.done, False, info

    def render(self):
        output = f"Player 1 Total: {self.player1_total}\n"
        output += f"Player 2 Total: {self.player2_total}\n"
        output += f"Current Player: Player {self.current_player}\n"
        if self.opponent_last_number is not None:
            output += f"Opponent's Last Number: {self.opponent_last_number}\n"
        else:
            output += "No moves have been made yet.\n"
        return output

    def valid_moves(self):
        # Define the valid moves based on opponent's last number
        last_opponent_number = self.opponent_last_number
        invalid_numbers = []
        if last_opponent_number is not None:
            invalid_numbers = [
                last_opponent_number - 1,
                last_opponent_number,
                last_opponent_number + 1,
            ]
        # Filter out numbers not in 1-9
        invalid_numbers = [n for n in invalid_numbers if 1 <= n <= 9]
        # Valid numbers are 1-9 excluding invalid_numbers
        valid_numbers = [n for n in range(1, 10) if n not in invalid_numbers]
        # Return valid actions (indices in action_space)
        valid_actions = [n - 1 for n in valid_numbers]  # Indices 0-8
        return valid_actions

    def _get_observation(self):
        # Observation from the perspective of the current player
        if self.current_player == 1:
            own_total = self.player1_total
            opponent_total = self.player2_total
        else:
            own_total = self.player2_total
            opponent_total = self.player1_total
        opponent_last_number = (
            self.opponent_last_number if self.opponent_last_number is not None else 0
        )
        observation = np.array(
            [own_total, opponent_total, opponent_last_number], dtype=np.int32
        )
        return observation
